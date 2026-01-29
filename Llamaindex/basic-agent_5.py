from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()
Settings.llm = Ollama(model="llama3")

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b    

def subtract_numbers(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        return "Error: Division by zero"
    return a / b

add_tool = FunctionTool.from_defaults(
    fn=add_numbers,
    name="add_numbers",
    description="Add two numbers together.",
)
subtract_tool = FunctionTool.from_defaults(
    fn=subtract_numbers,
    name="subtract_numbers",
    description="Subtract two numbers.",
)
multiply_tool = FunctionTool.from_defaults(
    fn=multiply_numbers,
    name="multiply_numbers",
    description="Multiply two numbers.",
)
divide_tool = FunctionTool.from_defaults(
    fn=divide_numbers,
    name="divide_numbers",
    description="Divide two numbers.",
)

agents = ReActAgent(
    name="Calculator Agent",
    tools=[add_tool, subtract_tool, multiply_tool, divide_tool],
    llm=Settings.llm,
    verbose=True,
)

import asyncio


async def main():
    response = await agents.run("What is 20 + (2 * 4) / (6-2 ) give me explanations also?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
