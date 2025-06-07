import os

import gradio as gr

from mcp.client.stdio import StdioServerParameters
from smolagents import InferenceClientModel, CodeAgent, ToolCollection, OpenAIServerModel, ToolCallingAgent
from smolagents.mcp_client import MCPClient


# Initialize the model
model = OpenAIServerModel(
    model_id="gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY"),
)

mcp_client = None
try:
    config = [
        # {"url": "https://abidlabs-mcp-tools.hf.space/gradio_api/mcp/sse"},
        # {"url": "https://tisan-das-mcp-sentiment.hf.space/gradio_api/mcp/sse"},
        {"url": "http://localhost:7860/gradio_api/mcp/sse"}  # Local server
    ]

    mcp_client = MCPClient(config)
    #MCPClient(
    #    {"url": "http://localhost:7860/gradio_api/mcp/sse"}
    #)
    # MCPClient.from_config_file("path/to/config.json")
    tools = mcp_client.get_tools()

    # model = InferenceClientModel(token=os.getenv("HUGGINGFACE_API_TOKEN"))
    # agent = CodeAgent(tools=[*tools], model=model)
    agent = ToolCallingAgent(tools=[*tools], model=model)

    demo = gr.ChatInterface(
        fn=lambda message, history: str(agent.run(message)),
        type="messages",
        examples=["Prime factorization of 68"],
        title="Agent with MCP Tools",
        description="This is a simple agent that uses MCP tools to answer questions.",
    )

    demo.launch()
finally:
    if mcp_client is not None:
        mcp_client.disconnect()

