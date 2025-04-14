import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()

# Initialize Tavily search tool from LangChain
search_tool = TavilySearchResults(api_key=os.environ.get("TAVILY_API_KEY"))


# Define the state for the agent
class SearchAgentState(TypedDict):
    query: str
    results: List[str]


# Define the function to perform the search
def perform_search(state: SearchAgentState) -> SearchAgentState:
    """Perform a search using Tavily search tool."""
    query = state["query"]
    search_results = search_tool.invoke({"query": query, "max_results": 3})

    # Check if search_results is a list of results or a single string
    if isinstance(search_results, list):
        results = [(result.get("content", ""),result.get("url",""),result.get("score",0)) for result in search_results]
    else:
        results = [search_results]

    return {"query": query, "results": results}


# Define the workflow
def build_graph():
    workflow = StateGraph(SearchAgentState)

    # Add nodes
    workflow.add_node("search", perform_search)

    # Define edges
    workflow.add_edge("search", END)

    # Set entry point
    workflow.set_entry_point("search")

    return workflow.compile()


# Main execution function
def search_with_agent(query: str):
    graph = build_graph()

    # Initial state
    initial_state = {"query": query, "results": []}

    # Execute the graph
    final_state = graph.invoke(initial_state)

    return final_state["results"]


if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    print(search_with_agent(user_query))