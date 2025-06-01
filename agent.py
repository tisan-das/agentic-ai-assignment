

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
    

from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel
import os

# Initialize Gemini model using OpenAI-compatible API
model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.GetEnv("KEY"),
)

# Create agent with Gemini model
agent = CodeAgent(tools=[], model=model)
