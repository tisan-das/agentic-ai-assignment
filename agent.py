

# # --- Basic Agent Definition ---
# # ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
# class BasicAgent:
#     def __init__(self):
#         print("BasicAgent initialized.")
#     def __call__(self, question: str) -> str:
#         print(f"Agent received question (first 50 chars): {question[:50]}...")
#         fixed_answer = "This is a default answer."
#         print(f"Agent returning fixed answer: {fixed_answer}")
#         return fixed_answer
    

from smolagents import CodeAgent, tool, DuckDuckGoSearchTool, OpenAIServerModel
import os

# Initialize Gemini model using OpenAI-compatible API
model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.GetEnv("KEY"),
)


from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}


# Create agent with Gemini model
agent = CodeAgent(tools=[
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    DuckDuckGoSearchTool(),
    arvix_search,
], model=model)
