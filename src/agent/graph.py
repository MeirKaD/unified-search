"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from typing import Any, Dict, TypedDict
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from agent.nodes import intent_classifier_node, google_search_node, web_unlocker_node , final_processing_node
import re
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str

class SearchState(TypedDict):
    """State definition for the unified search graph."""
    
    # Input
    query: str
    
    # Intent Classification Results
    intent: Optional[str]
    intent_confidence: Optional[float]
    intent_reasoning: Optional[str]
    search_strategy: Optional[str]
    
    # Search Results
    raw_results: List[Dict[str, Any]]
    final_results: Optional[List[Dict[str, Any]]]  # ADD THIS LINE
    query_summary: Optional[str]  # ADD THIS LINE
    total_processed: Optional[int]  # ADD THIS LINE
    # Node Completion Flags
    google_search_completed: Optional[bool]
    web_unlocker_completed: Optional[bool]
    final_processing_completed: Optional[bool]
    # Error Handling
    error: Optional[str]
    google_search_error: Optional[str]
    web_unlocker_error: Optional[str]
    final_processing_error: Optional[str]


def route_after_intent(state: Dict[str, Any]) -> Literal["google_search", "web_unlocker"]:
    """
    Routing function to determine next node after intent classification.
    
    Args:
        state: Current graph state with intent classification results
        
    Returns:
        Name of the next node to execute
    """
    # Check if specific URL is provided in query
    query = state.get("query", "")
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    has_url = bool(re.search(url_pattern, query))
    
    if has_url:
        return "web_unlocker"
    
    # Route based on search strategy from intent classification
    search_strategy = state.get("search_strategy", "google_only")
    
    if search_strategy == "web_unlocker_only":
        return "web_unlocker"
    elif search_strategy in ["google_only", "google_then_scrape", "both_parallel"]:
        return "google_search"
    else:
        # Fallback to Google search
        return "google_search"

def route_after_google(state: Dict[str, Any]) -> Literal["web_unlocker", "final_processing"]:
    """Route after Google search - continue to scraping if needed"""
    search_strategy = state.get("search_strategy", "google_only")
    
    if search_strategy in ["google_then_scrape", "both_parallel"]:
        return "web_unlocker"
    else:
        return "final_processing"

def route_after_web_unlocker(state: Dict[str, Any]) -> Literal["final_processing"]:
    """Route after web unlocker - always go to final processing"""
    return "final_processing"

def create_unified_search_graph() -> StateGraph:
    """
    Create the unified search graph with conditional routing.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize the graph
    workflow = StateGraph(SearchState)
    
    # Add nodes
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("google_search", google_search_node)
    workflow.add_node("web_unlocker", web_unlocker_node)
    workflow.add_node("final_processing", final_processing_node)
    
    # Set entry point
    workflow.add_edge(START, "intent_classifier")
    
    # Add conditional routing after intent classification
    workflow.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "google_search": "google_search",
            "web_unlocker": "web_unlocker"
        }
    )
    workflow.add_conditional_edges(
        "google_search",
        route_after_google,
        {
            "web_unlocker": "web_unlocker",
            "final_processing": "final_processing" 
        }
    )

    workflow.add_conditional_edges(
        "web_unlocker",
        route_after_web_unlocker,
        {
            "final_processing": "final_processing"
        }
    )

    workflow.add_edge("final_processing", END)
    return workflow.compile()

graph = create_unified_search_graph()