from typing import Dict, Any, Optional , List
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from mcp_use.client import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
load_dotenv()

class IntentClassification(BaseModel):
    """Intent classification result for search queries."""
    
    intent: str = Field(
        description="The classified intent: general_search, product_search, web_scraping, or comparison"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation for the classification decision"
    )
class ScoredResult(BaseModel):
    """Individual scored and sanitized search result."""
    title: str = Field(description="Cleaned title of the result")
    url: str = Field(description="Valid URL of the result")
    snippet: str = Field(description="Cleaned and relevant snippet")
    source: str = Field(description="Source of the result (google_search, web_unlocker)")
    relevance_score: float = Field(description="Relevance score between 0 and 1", ge=0.0, le=1.0)
    quality_score: float = Field(description="Quality score between 0 and 1", ge=0.0, le=1.0)
    final_score: float = Field(description="Combined final score between 0 and 1", ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(description="Additional metadata", default_factory=dict)

class FinalResults(BaseModel):
    """Final processed and scored results."""
    results: List[ScoredResult] = Field(description="List of scored results")
    total_processed: int = Field(description="Total number of results processed")
    query_summary: str = Field(description="Brief summary of what was found for the query")

def intent_classifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to classify search intent using LLM with structured output.
    
    Args:
        state: The current graph state containing the query
        
    Returns:
        Updated state with intent classification results
    """
    
    # Initialize LLM with structured output
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
    )
    
    structured_llm = llm.with_structured_output(IntentClassification)
    
    system_prompt = """You are an expert intent classifier for a unified search system.

Analyze the user query and classify it into one of these four categories:

1. **general_search**: General information queries like news, facts, definitions, how-to questions, explanations
2. **product_search**: Queries about products, shopping, prices, reviews, buying recommendations  
3. **web_scraping**: Queries that need data extraction from specific websites or require scraping
4. **comparison**: Queries comparing multiple items, services, or options

Provide your classification with confidence and reasoning.

Examples:
- "latest news about AI" → general_search
- "best laptops under $1000" → product_search  
- "extract contact info from linkedin.com" → web_scraping
- "iPhone vs Samsung Galaxy comparison" → comparison"""

    try:
        query = state.get("query", "")
        
        if not query:
            raise ValueError("No query found in state")
        
        # Create the full prompt
        full_prompt = f"{system_prompt}\n\nClassify this query: '{query}'"
        
        # Get structured output from LLM
        result = structured_llm.invoke(full_prompt)
        
        # Update state with classification results
        state["intent"] = result.intent
        state["intent_confidence"] = result.confidence
        state["intent_reasoning"] = result.reasoning
        
        # Set search strategy based on intent
        if result.intent == "general_search":
            state["search_strategy"] = "google_only"
        elif result.intent == "product_search":
            state["search_strategy"] = "google_then_scrape"
        elif result.intent == "web_scraping":
            state["search_strategy"] = "web_unlocker_only"
        elif result.intent == "comparison":
            state["search_strategy"] = "both_parallel"
        else:
            state["search_strategy"] = "google_only"  # fallback
        
        return state
        
    except Exception as e:
        # Handle errors gracefully with fallback values
        state["intent"] = "general_search"
        state["intent_confidence"] = 0.5
        state["intent_reasoning"] = f"Classification failed: {str(e)}"
        state["search_strategy"] = "google_only"
        state["error"] = f"Intent classification error: {str(e)}"
        
        return state
    
async def google_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to perform Google search using Bright Data's MCP search_engine.
    
    Args:
        state: The current graph state containing the query
        
    Returns:
        Updated state with Google search results
    """
    
    try:
        query = state.get("query", "")
        
        if not query:
            raise ValueError("No query found in state")
        
        # Configure Bright Data MCP client
        browserai_config = {
            "mcpServers": {
                "BrightData": {
                    "command": "npx",
                    "args": ["@brightdata/mcp"],
                    "env": {
                        "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                        "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                    }
                }
            }
        }
        
        client = MCPClient.from_dict(browserai_config)
        adapter = LangChainAdapter()
        tools = await adapter.create_tools(client)
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt="""You are a search agent that uses the search_engine tool to get Google search results.

Use the search_engine tool to search for the user's query and return clean, structured results.
Focus on getting the most relevant search results that answer the user's question.

Return the results in a structured format with:
- Title
- URL  
- Snippet/description
- Source information

Be precise and only return relevant search results."""
        )
        
        # Execute search through agent
        result = await agent.ainvoke({
            "messages": [{
                "role": "user", 
                "content": f"Search for: {query}. Please provide structured search results with title, URL, description and source information for each result."
            }]
        })
        
        # Extract search results from agent response
        search_response = result["messages"][-1].content
        
        # Use LLM with structured output to parse results
        google_results = await parse_search_results_structured(search_response, query)
        
        # Update state with Google search results
        if not state.get("raw_results"):
            state["raw_results"] = []
        
        state["raw_results"].extend(google_results)
        state["google_search_completed"] = True
        
        return state
        
    except Exception as e:
        # Handle errors gracefully
        state["google_search_error"] = f"Google search failed: {str(e)}"
        if not state.get("raw_results"):
            state["raw_results"] = []
        
        return state


from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """Individual search result structure."""
    title: str = Field(description="The title of the search result")
    url: str = Field(description="The URL of the search result") 
    snippet: str = Field(description="The description or snippet of the search result")

class SearchResultsList(BaseModel):
    """List of structured search results."""
    results: List[SearchResult] = Field(description="List of search results found")
    total_found: int = Field(description="Total number of results found")

async def parse_search_results_structured(search_response: str, query: str) -> List[Dict[str, Any]]:
    """
    Parse search results using LLM with structured output for precision.
    
    Args:
        search_response: Raw response from the search agent
        query: Original search query
        
    Returns:
        List of structured search results
    """
    try:
        # Create LLM with structured output for parsing
        parsing_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_parsing_llm = parsing_llm.with_structured_output(SearchResultsList)
        
        parsing_prompt = f"""
        Parse the following search results response into a structured format.
        Extract the title, URL, and snippet/description for each search result found.
        
        Original query: {query}
        
        Search results response:
        {search_response}
        
        Return a structured list of search results with title, url, and snippet for each result.
        If URLs are missing or incomplete, leave them as empty strings.
        If descriptions are missing, use "No description available".
        """
        
        parsed_results = await structured_parsing_llm.ainvoke(parsing_prompt)
        
        # Convert to standard format
        formatted_results = []
        for result in parsed_results.results:
            formatted_results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": "google_search",
                "metadata": {
                    "search_engine": "google",
                    "via": "bright_data_mcp",
                    "query": query
                }
            })
        
        return formatted_results[:10]  # Limit to top 10
        
    except Exception as e:
        # Fallback: create one result with the raw response
        return [{
            "title": f"Search results for: {query}",
            "url": "",
            "snippet": search_response[:200] + "..." if len(search_response) > 200 else search_response,
            "source": "google_search",
            "metadata": {
                "parsing_error": str(e),
                "raw_response": True,
                "query": query
            }
        }]
    
class ScrapedResult(BaseModel):
    """Individual scraped result structure."""
    title: str = Field(description="The title or name of the scraped item")
    url: str = Field(description="The URL where this information was found")
    snippet: str = Field(description="Key information or description scraped from the page")
    price: str = Field(description="Price if this is a product, otherwise empty string", default="")
    rating: str = Field(description="Rating or review score if available, otherwise empty string", default="")
    availability: str = Field(description="Availability status if applicable, otherwise empty string", default="")


class ScrapedResultsList(BaseModel):
    """List of structured scraped results."""
    results: List[ScrapedResult] = Field(description="List of scraped results found")
    total_found: int = Field(description="Total number of results found")
    source_website: str = Field(description="The main website that was scraped")


async def web_unlocker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function to perform web scraping using Bright Data's Web Unlocker.
    
    Args:
        state: The current graph state containing the query
        
    Returns:
        Updated state with scraped results
    """
    
    try:
        query = state.get("query", "")
        
        if not query:
            raise ValueError("No query found in state")
        
        # Configure Bright Data MCP client
        browserai_config = {
            "mcpServers": {
                "BrightData": {
                    "command": "npx",
                    "args": ["@brightdata/mcp"],
                    "env": {
                        "API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN"),
                        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE", "unblocker"),
                        "BROWSER_ZONE": os.getenv("BROWSER_ZONE", "scraping_browser")
                    }
                }
            }
        }
        
        # Initialize MCP client and tools
        client = MCPClient.from_dict(browserai_config)
        adapter = LangChainAdapter()
        tools = await adapter.create_tools(client)
        
        # Create agent with Bright Data tools
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt="""You are a web scraping agent that uses Bright Data tools to extract specific information from websites.

You MUST use the scraping tools to actually scrape the provided URLs. Use scrape_as_markdown for general websites or web_data_amazon for Amazon URLs.

IMPORTANT: Don't just analyze the instructions - actually execute the scraping tools with the URLs provided.

For product queries, focus on:
- Product names/titles
- Prices
- Ratings/reviews
- Availability
- Product URLs

For general scraping, focus on:
- Relevant content titles
- Key information snippets
- Source URLs

Be precise and extract only relevant information that matches the user's query."""
        )
        
        # Determine target URL or scraping strategy based on query
        scraping_instruction = await generate_scraping_instruction(query,state)
        
        # Execute scraping through agent
        result = await agent.ainvoke({
            "messages": [{
                "role": "user", 
                "content": scraping_instruction
            }]
        })
        
        # Extract scraping results from agent response
        scraping_response = result["messages"][-1].content
        
        # Use LLM with structured output to parse results
        scraped_results = await parse_scraped_results_structured(scraping_response, query)
        
        # Update state with scraped results
        if not state.get("raw_results"):
            state["raw_results"] = []
        
        state["raw_results"].extend(scraped_results)
        state["web_unlocker_completed"] = True
        
        return state
        
    except Exception as e:
        # Handle errors gracefully
        state["web_unlocker_error"] = f"Web scraping failed: {str(e)}"
        if not state.get("raw_results"):
            state["raw_results"] = []
        
        return state


async def generate_scraping_instruction(query: str, state: Dict[str, Any]) -> str:
    """
    Generate specific scraping instructions based on the query using LLM.
    
    Args:
        query: The user's search query
        state: Current state containing Google search results
        
    Returns:
        Detailed instruction for the scraping agent
    """
    raw_results = state.get("raw_results", [])
    google_urls = [r.get("url", "") for r in raw_results if r.get("source") == "google_search" and r.get("url")]
    
    instruction_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    if google_urls:
        instruction_prompt = f"""
        Analyze this user query and generate specific web scraping instructions for a Bright Data agent.
        
        The user searched for: {query}
        Google search found these relevant URLs: {google_urls[:5]}  # Top 5 URLs
        
        Available tools:
        - web_data_amazon: For Amazon product data
        - scrape_as_markdown: General webpage content
        - scrape_as_html: Raw HTML extraction
        - Other structured extractors for major sites
        
        Generate detailed instructions that specify:
        1. Which tool(s) to use for each URL (web_data_amazon for Amazon URLs, scrape_as_markdown for others)
        2. Extract specific product information:
           - Individual product names and models
           - Exact prices
           - Customer ratings/reviews
           - Stock availability
           - Product specifications
        3. Focus on finding actual products that match the user's criteria, not just search result pages
        
        Return clear, actionable instructions for the scraping agent.
        """
    else:
        instruction_prompt = f"""
        Analyze this user query and generate specific web scraping instructions for a Bright Data agent.
        
        User query: {query}
        
        Available tools:
        - web_data_amazon: For Amazon product data
        - scrape_as_markdown: General webpage content
        - scrape_as_html: Raw HTML extraction
        - Other structured extractors for major sites
        
        Generate detailed instructions that specify:
        1. Which tool(s) to use
        2. What specific information to extract
        3. Target URL if determinable
        4. Focus areas based on query intent
        
        Return clear, actionable instructions for the scraping agent.
        """
    
    instruction_response = await instruction_llm.ainvoke(instruction_prompt)
    return instruction_response.content


async def parse_scraped_results_structured(scraping_response: str, query: str) -> List[Dict[str, Any]]:
    """
    Parse scraped results using LLM with structured output for precision.
    
    Args:
        scraping_response: Raw response from the scraping agent
        query: Original search query
        
    Returns:
        List of structured scraped results
    """
    try:
        # Create LLM with structured output for parsing
        parsing_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_parsing_llm = parsing_llm.with_structured_output(ScrapedResultsList)
        
        parsing_prompt = f"""
        Parse the following scraped content into a structured format.
        Extract relevant information that matches the user's query.
        
        Original query: {query}
        
        Scraped content:
        {scraping_response}
        
        For each relevant item found, extract:
        - title: Product name, article title, or content title
        - url: Source URL if available
        - snippet: Key information or description
        - price: If it's a product with pricing
        - rating: If reviews/ratings are available
        - availability: If availability status is mentioned
        
        Focus only on results that are relevant to the user's query.
        If no specific items are found, create one result summarizing the scraped content.
        """
        
        parsed_results = await structured_parsing_llm.ainvoke(parsing_prompt)
        
        # Convert to standard format
        formatted_results = []
        for result in parsed_results.results:
            formatted_results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": "web_unlocker",
                "metadata": {
                    "scraping_method": "bright_data_unlocker",
                    "query": query,
                    "price": result.price,
                    "rating": result.rating,
                    "availability": result.availability,
                    "source_website": parsed_results.source_website
                }
            })
        
        return formatted_results[:10]  # Limit to top 10
        
    except Exception as e:
        # Fallback: create one result with the raw response
        return [{
            "title": f"Scraped content for: {query}",
            "url": "",
            "snippet": scraping_response[:200] + "..." if len(scraping_response) > 200 else scraping_response,
            "source": "web_unlocker",
            "metadata": {
                "parsing_error": str(e),
                "raw_response": True,
                "query": query
            }
        }]
    
async def final_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final node to sanitize, score, and rank search results.
    
    Args:
        state: The current graph state containing raw results
        
    Returns:
        Updated state with final processed results
    """
    
    try:
        raw_results = state.get("raw_results", [])
        query = state.get("query", "")
        max_results = state.get("max_results", 5)  # Default to 5 results
        
        if not raw_results:
            state["final_results"] = []
            state["final_processing_completed"] = True
            return state
        
        # Initialize LLM for scoring and processing
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        structured_llm = llm.with_structured_output(FinalResults)
        
        # Prepare raw results for processing
        results_text = ""
        for i, result in enumerate(raw_results, 1):
            results_text += f"""
Result {i}:
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Snippet: {result.get('snippet', 'N/A')}
Source: {result.get('source', 'N/A')}
Metadata: {result.get('metadata', {})}
---
"""
        
        processing_prompt = f"""
You are an expert result processor and scorer. Process these search results for the query: "{query}"

Your tasks:
1. **Sanitize**: Clean titles and snippets, validate URLs, remove duplicates
2. **Score**: Rate each result on relevance (0-1) and quality (0-1)
3. **Rank**: Calculate final_score = (relevance_score * 0.7) + (quality_score * 0.3)
4. **Select**: Return the top {max_results} results

Scoring criteria:
- **Relevance**: How well does this result answer the user's query?
- **Quality**: Is the source authoritative? Is the information clear and useful?

Raw results to process:
{results_text}

Requirements:
- Only include results with valid URLs (not empty strings)
- Remove obvious duplicates (same URL or very similar content)
- Prioritize authoritative sources (official websites, established publications)
- Ensure titles and snippets are clean and informative
- Provide a brief query_summary explaining what was found

Return the top {max_results} scored and ranked results.
"""
        
        # Get structured output from LLM
        processed_results = await structured_llm.ainvoke(processing_prompt)
        
        # Convert to final format
        final_results = []
        for result in processed_results.results:
            # Find original metadata from raw results
            original_metadata = {}
            for raw_result in raw_results:
                if (raw_result.get('title', '') == result.title or 
                    raw_result.get('url', '') == result.url):
                    original_metadata = raw_result.get('metadata', {})
                    break
            
            final_results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
                "relevance_score": result.relevance_score,
                "quality_score": result.quality_score,
                "final_score": result.final_score,
                "metadata": {**original_metadata, **result.metadata}
            })
        
        # Update state with final results
        state["final_results"] = final_results
        state["query_summary"] = processed_results.query_summary
        state["total_processed"] = processed_results.total_processed
        state["final_processing_completed"] = True
        
        return state
        
    except Exception as e:
        # Handle errors gracefully - return raw results as fallback
        state["final_processing_error"] = f"Final processing failed: {str(e)}"
        
        # Create simple fallback results
        raw_results = state.get("raw_results", [])
        max_results = state.get("max_results", 5)
        
        fallback_results = []
        for i, result in enumerate(raw_results[:max_results]):
            fallback_results.append({
                "title": result.get('title', 'N/A'),
                "url": result.get('url', 'N/A'),
                "snippet": result.get('snippet', 'N/A'),
                "source": result.get('source', 'N/A'),
                "relevance_score": 0.5,  # Default score
                "quality_score": 0.5,    # Default score
                "final_score": 0.5,      # Default score
                "metadata": result.get('metadata', {})
            })
        
        state["final_results"] = fallback_results
        state["query_summary"] = f"Found {len(fallback_results)} results for: {state.get('query', '')}"
        state["final_processing_completed"] = True
        
        return state