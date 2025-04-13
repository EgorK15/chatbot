import os
import json
from typing import TypedDict, List, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_community import GmailToolkit
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import email

# Load environment variables
load_dotenv()

# Configure the LLM
model = ChatOpenAI(
    model_name=os.environ.get("MODEL_NAME"),
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE"),
    temperature=0
)

# Initialize Gmail toolkit
toolkit = GmailToolkit(credentials_path="credentials.json")
gmail_tools = toolkit.get_tools()

# Define our state
class AgentState(TypedDict):
    messages: List[Annotated[dict, "Messages being passed around"]]
    query: str
    email_data: List[dict]
    answer: str
    sources: List[str]
    answer_bool: bool
class ChatResponse(BaseModel):
    answer: bool
    answer_txt: str = Field(description="Ответ на вопрос пользователя")
    sources: list = Field(description="Источники информации, использованные для ответа", default_factory=list)

# Define custom tools
@tool
def search_emails(query: str) -> str:
    """Search the last 10 emails for information related to the query"""
    search_tool = [tool for tool in gmail_tools if tool.name == "search_gmail"]

    # Search for the most recent 10 emails
    search_results = search_tool[0].invoke({"query": "is:inbox", "max_results": 10})
    return json.dumps(search_results)

@tool
def get_email_content(message_id: str) -> str:
    """Get the content of a specific email by message ID"""
    get_tool = next((tool for tool in gmail_tools if tool.name == "get_gmail_message"), None)
    if not get_tool:
        return "Gmail get tool not available"

    email_content = get_tool.invoke({"message_id": message_id})
    return json.dumps(email_content)

# Define nodes for our graph
def search_node(state: AgentState) -> AgentState:
    """Node to search for emails"""
    search_results = json.loads(search_emails(state["query"]))
    email_data = []

    for message in search_results:
        if message and "id" in message:
            email_content = json.loads(get_email_content(message["id"]))
            email_data.append(email_content)

    return {"messages": state["messages"], "query": state["query"], "email_data": email_data, "answer": "", "sources": []}

def analyze_node(state: AgentState) -> AgentState:
    """Node to analyze email content and generate an answer"""
    messages = state["messages"].copy()

    # Create prompt with email data
    email_summaries = []
    sources = []
    for i, emaill in enumerate(state["email_data"]):
        sender = ''.join(part.decode(encoding or 'utf-8') if isinstance(part, bytes) else part for part, encoding in email.header.decode_header(emaill.get("sender", "Unknown")))
        subject = ''.join(part.decode(encoding or 'utf-8') if isinstance(part, bytes) else part for part, encoding in email.header.decode_header(emaill.get("subject", "Unknown")))
        body = " ".join(emaill.get("body","no body").replace("\xa0", " ").replace("\n", " ").strip().split())
        body = body.replace("  ","")

        summary = f"Email {i+1}:\nFrom: {sender}\nSubject: {subject}\nContent: {body}\n"
        email_summaries.append(summary)
        idd = emaill.get("id", "Unknown")
        sources.append(f"{sender} - {subject} - {idd}")

    # Create system message
    enter = "\n\n"
    system_message = SystemMessage(content=f"""
    You are an email information retrieval assistant. Check the following emails for information related to the user's query. If there is information that answers the query, summarize it. If not, say "No relevant information found".
    Your output must be in JSON  format:  answer (1 or 0), answer_txt, sources (numbers of emails).
    User Query: {state["query"]}
    
    Email Data:
    {enter.join(email_summaries)}
    """)

    # Add our messages to the state
    messages.append(system_message)

    # Get response from LLM
    structured_llm = model.with_structured_output(ChatResponse,method="json_mode")
    response = structured_llm.invoke(messages)
    answer_bool = response.answer
    answer = response.answer_txt
    sources_true = []
    for i in response.sources:
        sources_true.append(sources[i-1])
    return {
        "messages": state["messages"],
        "query": state["query"],
        "email_data": state["email_data"],
        "answer": answer,
        "sources": sources_true,
        "answer_bool": bool(answer_bool)
    }

# Define our workflow
def build_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search_emails", search_node)
    workflow.add_node("analyze_emails", analyze_node)

    # Define edges
    workflow.add_edge("search_emails", "analyze_emails")
    workflow.add_edge("analyze_emails", END)

    # Set entry point
    workflow.set_entry_point("search_emails")

    return workflow.compile()

# Main execution function
def search_emails_for_info(query: str):
    graph = build_graph()

    # Initial state
    initial_state = {
        "messages": [],
        "query": query,
        "email_data": [],
        "answer": "",
        "sources": [],
        "answer_bool":False
    }

    # Execute the graph
    final_state = graph.invoke(initial_state)

    # Format results
    return {"answer_bool" : final_state["answer_bool"],"answer": final_state["answer"],"messages": final_state["sources"]}

if __name__ == "__main__":
    query = input("Enter your search query for emails: ")
    print(search_emails_for_info(query))