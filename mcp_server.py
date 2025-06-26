from fastmcp import FastMCP
from src.agent.graph import create_unified_search_graph  
import asyncio

graph = create_unified_search_graph()
mcp = FastMCP("API-Documentation-Agent")

@mcp.tool()
async def search(search_term: str) -> str:
    """Run Unified search across multiple wesites and platforms"""""
    
    initial_state = {
 # Input
    "query": search_term,
    "intent": "",
    "intent_confidence": "",
    "intent_reasoning":"",
    "search_strategy":"",
    "raw_results":"",
    "final_results": "",
    "query_summary":"",
    "total_processed":"",
    "google_search_completed": "",
    "web_unlocker_completed": "",
    "final_processing_completed": "",
    "error":"",
    "google_search_error": "",
    "web_unlocker_error": "",
    "final_processing_error":""
    }
    
    try:
        result = await graph.ainvoke(initial_state)
        return result.get("final_results", "No results found.")
    except Exception as e:
        return f"Error searching {search_term} : {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")