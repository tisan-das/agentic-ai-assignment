

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
    

from smolagents import CodeAgent,ToolCallingAgent, tool, DuckDuckGoSearchTool, OpenAIServerModel,VisitWebpageTool
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
            model_id="gemini-2.5-flash-preview-05-20",
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

model = SimpleRateLimitedModel(requests_per_minute=3)

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.
    Args:
        a: first float
        b: second float
    """
    return a * b

@tool
def add(a: float, b: float) -> float:
    """Add two numbers.
    
    Args:
        a: first float
        b: second float
    """
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers.
    
    Args:
        a: first float
        b: second float
    """
    return a - b

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers.
    
    Args:
        a: first float
        b: second float
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
    """Search Wikipedia for a query and return maximum 10 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=10).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 10 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=10).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}


import pandas as pd

@tool
def load_excel_file(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """Load an Excel file and returns pandas dataframe containing the excel sheet data.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Optional sheet name to load (defaults to first sheet)
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        info = f"Loaded Excel file with {df.shape[0]} rows and {df.shape[1]} columns.\n"
        info += f"Columns: {list(df.columns)}\n"
        info += f"First 5 rows:\n{df.head().to_string()}"
        return df
    except Exception as e:
        return f"Error loading Excel file: {str(e)}"


from smolagents import SpeechToTextTool

@tool
def transcribe_audio_to_text(file_path: str) -> str:
    """Convert speech in audio file to text string.
    
    Args:
        file_path: Path to the audio file containing speech
    """
    try:
        # This would use a speech-to-text model like Whisper
        import whisper
        
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        
        return f"Transcribed text: {result['text']}"
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

import base64

@tool
def encode_image_for_analysis(image_path: str) -> str:
    """Encode image to base64 for processing by vision models.
    
    Args:
        image_path: Path to the image file
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return f"Image encoded successfully (length: {len(encoded_string)} characters)"
    except Exception as e:
        return f"Error encoding image: {str(e)}"
    

import chess
import chess.engine

@tool
def analyze_chess_position(fen: str, depth: int = 20, num_moves: int = 3) -> str:
    """Analyze a chess position using Stockfish engine.
    
    Args:
        fen: FEN string representing the chess position
        depth: Analysis depth (default 20)
        num_moves: Number of best moves to return (default 3)
    """
    try:
        # Initialize Stockfish engine
        engine = chess.engine.SimpleEngine.popen_uci("/usr/local/bin/stockfish")
        
        board = chess.Board(fen)
        
        # Analyze position with multiple principal variations
        search_limit = chess.engine.Limit(depth=depth)
        infos = engine.analyse(board, search_limit, multipv=num_moves)
        
        analysis = f"Chess Position Analysis (Depth {depth}):\n"
        analysis += f"Position: {fen}\n"
        analysis += f"Turn: {'White' if board.turn else 'Black'}\n\n"
        
        # Format each variation
        for i, info in enumerate(infos, 1):
            score = info["score"].white()  # Always from White's perspective
            pv = info["pv"]  # Principal variation (best moves)
            
            analysis += f"Move #{i}:\n"
            analysis += f"  Best move: {pv[0] if pv else 'None'}\n"
            analysis += f"  Evaluation: {score}\n"
            analysis += f"  Principal variation: {' '.join(str(move) for move in pv[:5])}\n\n"
        
        engine.quit()
        return analysis
        
    except Exception as e:
        return f"Error analyzing position: {str(e)}"
    

import yaml
with open("prompt.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)


# Create web agent and manager agent structure
code_agent = CodeAgent(tools=[
        multiply,
        add,
        subtract,
        divide,
        modulus,
        wiki_search,
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        arvix_search,
        load_excel_file,
        transcribe_audio_to_text,
        SpeechToTextTool(),
        encode_image_for_analysis,
        analyze_chess_position,
    ],
    model=model,
    max_steps=25,  # Allow up to 25 reasoning steps
    add_base_tools=True,
    # sandbox=E2BSandbox(),
    additional_authorized_imports=[
        "numpy",
        "requests",
        "markdownify",
        "re",
        "math",
        "json",
        "datetime",
        "urllib",
        "base64",
        "pymupdf",
        "fitz",
        "pandas", 
        "matplotlib",
        "seaborn",
        "openpyxl",
        "whisper",
        "librosa",
        "base64",
        "chess",
        "chess.engine",
    ],
    name = "code_agent",
    description = "An agent used for coding related activities, where the problem can be coded in python code"
)


node_code_agent = ToolCallingAgent(tools=[
        multiply,
        add,
        subtract,
        divide,
        modulus,
        wiki_search,
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        arvix_search,
        load_excel_file,
        transcribe_audio_to_text,
        SpeechToTextTool(),
        encode_image_for_analysis,
        analyze_chess_position,
    ],
    model=model,
    # system_prompt=SYSTEM_PROMPT,
    max_steps=25,  # Allow up to 25 reasoning steps
    add_base_tools=True,
    # sandbox=E2BSandbox(),
    # additional_authorized_imports=[
    #     "numpy",
    #     "requests",
    #     "markdownify",
    #     "re",
    #     "math",
    #     "json",
    #     "datetime",
    #     "urllib",
    #     "base64",
    # ],
    name = "node_code_agent",
    description = "An agent used for all the non-coding related activities"
)


agent = ToolCallingAgent(tools=[
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
    # system_prompt=SYSTEM_PROMPT,
    max_steps=25,  # Allow up to 25 reasoning steps
    add_base_tools=True,
    # sandbox=E2BSandbox(),
    # additional_authorized_imports=[
    #     "numpy",
    #     "requests",
    #     "markdownify",
    #     "re",
    #     "math",
    #     "json",
    #     "datetime",
    #     "urllib",
    #     "base64",
    # ],
    managed_agents = [code_agent, node_code_agent],
    name="sample_agent",
    prompt_templates=prompt_templates
)
