

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
    

from smolagents import CodeAgent, tool, DuckDuckGoSearchTool, OpenAIServerModel,VisitWebpageTool
# from smolagents import E2BSandbox
import os, time
from datetime import datetime

from langchain_core.rate_limiters import InMemoryRateLimiter

# Create rate limiter: 
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5/60,
    check_every_n_seconds=1,    # Check every second
    max_bucket_size=1           # Allow only 1 request in bucket
)


# # Initialize Gemini model using OpenAI-compatible API
# model = OpenAIServerModel(
#     model_id="gemini-2.0-flash",
#     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key=os.getenv("KEY"),
#     rate_limiter=rate_limiter
# )

class SimpleRateLimitedModel:
    """Simple, reliable rate-limited model with proper method proxying"""
    
    def __init__(self, requests_per_minute=5):
        print(f"🔧 Initializing Gemini model with {requests_per_minute} req/min limit")
        
        self.base_model = OpenAIServerModel(
            model_id="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("KEY"),
        )
        
        self.min_interval = 60.0 / requests_per_minute
        self.last_call_time = 0
        self.call_count = 0
        
    def __call__(self, messages, **kwargs):
        return self._rate_limited_call(lambda: self.base_model(messages, **kwargs))
    
    def generate(self, messages, **kwargs):
        """Proxy the generate method with rate limiting"""
        return self._rate_limited_call(lambda: self.base_model.generate(messages, **kwargs))
    
    def _rate_limited_call(self, func):
        """Apply rate limiting to any function call"""
        self.call_count += 1
        
        # Rate limiting logic
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        
        if self.last_call_time > 0 and elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"🚦 Call #{self.call_count}: Rate limit active")
            print(f"⏰ Waiting {wait_time:.1f}s ({datetime.now().strftime('%H:%M:%S')})")
            time.sleep(wait_time)
        
        # Make the API call
        self.last_call_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"🚀 Call #{self.call_count}: API request at {timestamp}")
        
        try:
            result = func()
            print(f"✅ Call #{self.call_count}: Success")
            return result
        except Exception as e:
            print(f"❌ Call #{self.call_count}: Error - {e}")
            raise
    
    def __getattr__(self, name):
        """Proxy any other methods to the base model"""
        attr = getattr(self.base_model, name)
        if callable(attr):
            return lambda *args, **kwargs: self._rate_limited_call(lambda: attr(*args, **kwargs))
        return attr

model = SimpleRateLimitedModel(requests_per_minute=5)

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
        VisitWebpageTool(),
        arvix_search,
    ],
    model=model,
    max_steps=25,  # Allow up to 15 reasoning steps
    add_base_tools=True,
    # sandbox=E2BSandbox(),
    additional_authorized_imports=[
        "numpy",
        "requests",
        "markdownify",
        "re",
        "math",
    ]
)
