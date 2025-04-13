import os
from typing import TypedDict, List, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure the LLM
model = ChatOpenAI(
    model_name=os.environ.get("MODEL_NAME"),
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE"),
    temperature=0
)

# Define our state
class ReflectionAgentState(TypedDict):
    messages: List[Annotated[dict, "Messages being passed around"]]
    query: str
    answer: str
    evaluation: str
    reflection: str
    attempts: int
    passed_evaluation: bool

# Define nodes for our graph
def generate_answer(state: ReflectionAgentState) -> ReflectionAgentState:
    """Generate an answer to the user's query"""
    messages = state["messages"].copy()

    system_message = SystemMessage(content="""
    You are a helpful AI assistant. Provide clear, accurate, and concise answers to user questions.
    """)

    user_message = HumanMessage(content=state["query"])

    # Add messages
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.append(system_message)

    # Only add user message if it's not already added
    if not any(isinstance(msg, HumanMessage) and msg.content == state["query"] for msg in messages):
        messages.append(user_message)

    # Generate answer
    response = model.invoke(messages)

    return {
        "messages": messages + [response],
        "query": state["query"],
        "answer": response.content,
        "evaluation": state.get("evaluation", ""),
        "reflection": state.get("reflection", ""),
        "attempts": state.get("attempts", 0) + 1,
        "passed_evaluation": False
    }

def evaluate_answer(state: ReflectionAgentState) -> ReflectionAgentState:
    """Evaluate the generated answer for quality"""
    evaluation_prompt = SystemMessage(content=f"""
    You are an evaluator assessing the quality of an AI response.
    
    User Query: {state["query"]}
    
    AI Response: {state["answer"]}
    
    Evaluate the response based on the following criteria:
    1. Accuracy: Is the information correct?
    2. Completeness: Does it fully answer the query?
    3. Clarity: Is the response clear and well-structured?
    4. Conciseness: Is the response appropriately concise?
    
    First, analyze each criterion thoroughly.
    Then, decide if the response PASSES or FAILS the evaluation.
    If it FAILS, explain why specifically and what needs improvement.
    
    Your evaluation format:
    Accuracy: [analysis]
    Completeness: [analysis]
    Clarity: [analysis]
    Conciseness: [analysis]
    
    Verdict: [PASS/FAIL]
    Improvement Areas: [only if FAIL]
    """)

    evaluation_result = model.invoke([evaluation_prompt])

    # Check if the evaluation contains PASS
    passed = "PASS" in evaluation_result.content and "FAIL" not in evaluation_result.content

    return {
        "messages": state["messages"],
        "query": state["query"],
        "answer": state["answer"],
        "evaluation": evaluation_result.content,
        "reflection": state.get("reflection", ""),
        "attempts": state["attempts"],
        "passed_evaluation": passed
    }

def reflect_and_improve(state: ReflectionAgentState) -> ReflectionAgentState:
    """Reflect on evaluation feedback to improve the answer"""
    reflection_prompt = SystemMessage(content=f"""
    You are a reflective AI focused on self-improvement.
    
    User Query: {state["query"]}
    
    Your previous response: {state["answer"]}
    
    Evaluation feedback: {state["evaluation"]}
    
    Based on this feedback, reflect on what specifically needs improvement in your response.
    Be concrete and specific in your reflection.
    """)

    reflection_result = model.invoke([reflection_prompt])

    return {
        "messages": state["messages"],
        "query": state["query"],
        "answer": state["answer"],
        "evaluation": state["evaluation"],
        "reflection": reflection_result.content,
        "attempts": state["attempts"],
        "passed_evaluation": state["passed_evaluation"]
    }

def revise_answer(state: ReflectionAgentState) -> ReflectionAgentState:
    """Generate an improved answer based on reflection"""
    revision_prompt = SystemMessage(content=f"""
    You are a helpful AI assistant. Generate an improved response based on the following:
    
    User Query: {state["query"]}
    
    Your previous response: {state["answer"]}
    
    Evaluation feedback: {state["evaluation"]}
    
    Your reflection: {state["reflection"]}
    
    Provide a new, improved response that addresses all the issues mentioned in the evaluation.
    """)

    revised_response = model.invoke([revision_prompt])

    return {
        "messages": state["messages"],
        "query": state["query"],
        "answer": revised_response.content,
        "evaluation": state["evaluation"],
        "reflection": state["reflection"],
        "attempts": state["attempts"],
        "passed_evaluation": False
    }

# Decision making function
def should_continue_refining(state: ReflectionAgentState) -> Literal["revise", "end"]:
    """Decide whether to continue refining or end the process"""
    # End if passed evaluation or reached max attempts
    if state["passed_evaluation"] or state["attempts"] >= 3:
        return "end"
    return "revise"

# Define our workflow
def build_graph():
    workflow = StateGraph(ReflectionAgentState)

    # Add nodes
    workflow.add_node("generate", generate_answer)
    workflow.add_node("evaluate", evaluate_answer)
    workflow.add_node("reflect", reflect_and_improve)
    workflow.add_node("revise", revise_answer)

    # Define edges
    workflow.add_edge("generate", "evaluate")
    # Fix for conditional routing - use the function directly
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_refining,
        {
            "revise": "reflect",
            "end": END
        }
    )
    workflow.add_edge("reflect", "revise")
    workflow.add_edge("revise", "evaluate")

    # Set entry point
    workflow.set_entry_point("generate")

    return workflow.compile()

# Main execution function
def get_reflective_answer(query: str):
    graph = build_graph()

    # Initial state
    initial_state = {
        "messages": [],
        "query": query,
        "answer": "",
        "evaluation": "",
        "reflection": "",
        "attempts": 0,
        "passed_evaluation": False
    }

    # Execute the graph
    final_state = graph.invoke(initial_state)

    # Format results
    print("\n--- Reflection Agent Results ---")
    print(f"Query: {query}")
    print(f"\nFinal Answer (After {final_state['attempts']} attempts):")
    print(final_state["answer"])
    print("\nFinal Evaluation:")
    print(final_state["evaluation"])

    return final_state["answer"]

if __name__ == "__main__":
    query = input("Enter your question: ")
    get_reflective_answer(query)