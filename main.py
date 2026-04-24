import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Import modular components
from src.rag_pipeline import build_rag_pipeline, generate_answer
from src.llm import get_llm

load_dotenv()

# State definition
class GraphState(TypedDict):
    query: str
    category: str
    response: str

# Nodes
def categorizer(state: GraphState):
    """
    Analyzes the user query to determine the path.
    Improved prompt to prevent technical questions from being treated as greetings.
    """
    llm = get_llm("llama-3.3-70b-versatile")
    
    system_prompt = (
        "You are an intent classifier. Categorize the user query into exactly one of these labels: 'greeting', 'support', or 'escalate'.\n"
        "- 'greeting': Only for 'hi', 'hello', 'good morning', etc.\n"
        "- 'support': For any question about data, analytics, technology, or specific information lookup.\n"
        "- 'escalate': For complaints, requests for humans, or complex emotional queries.\n"
        "Return ONLY the word."
    )
    
    res = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state['query']}
    ]).content.lower()
    
    # Precise mapping logic
    category = "support" # Default to support to be safe
    if "greeting" in res: 
        # Double check it's not actually a question
        if "?" in state['query'] or len(state['query'].split()) > 3:
            category = "support"
        else:
            category = "greeting"
    elif "escalate" in res: 
        category = "escalate"
        
    return {"category": category}

def rag_agent(state: GraphState):
    """
    The RAG Agent node. It uses the file path to find context.
    """
    # Ensure this path is correct on your local machine
    pdf_path = r"C:\Users\vines\OneDrive\Pictures\Desktop\RAG-Based Customer Support Assistant\data\Owners_Manual.pdf"
    
    try:
        vs = build_rag_pipeline(pdf_path)
        answer = generate_answer(vs, state['query'])
        
        # If the RAG system returns 'ESCALATE', we update the category
        category = state['category']
        if "ESCALATE" in answer.upper():
            category = "escalate"
            
        return {"response": answer, "category": category}
    except Exception as e:
        return {"response": f"Error in RAG Pipeline: {str(e)}", "category": "escalate"}

def escalator(state: GraphState):
    return {"response": "[HITL] I am unable to find a specific answer in my database. I have forwarded your query '"+ state['query'] +"' to our human data analysts."}

def greeter(state: GraphState):
    return {"response": "Hello! I am your Data Analytics Assistant. How can I help you today?"}

# Graph Construction
workflow = StateGraph(GraphState)
workflow.add_node("categorizer", categorizer)
workflow.add_node("rag_agent", rag_agent)
workflow.add_node("escalator", escalator)
workflow.add_node("greeter", greeter)

workflow.set_entry_point("categorizer")

def router(state):
    if state["category"] == "greeting": return "greeting"
    if state["category"] == "escalate": return "escalate"
    return "support"

workflow.add_conditional_edges("categorizer", router, {
    "greeting": "greeter",
    "support": "rag_agent",
    "escalate": "escalator"
})

workflow.add_conditional_edges("rag_agent", lambda x: "escalate" if x["category"] == "escalate" else "end", {
    "escalate": "escalator",
    "end": END
})

workflow.add_edge("greeter", END)
workflow.add_edge("escalator", END)

app = workflow.compile()

if __name__ == "__main__":
    print("\n--- RAG Customer Support Assistant Active ---")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break
            
        for output in app.stream({"query": query}):
            for key, value in output.items():
                if "response" in value:
                    print(f"\nAssistant [{key}]: {value['response']}")