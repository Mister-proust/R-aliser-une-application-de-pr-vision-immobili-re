import os
import asyncio
import logging
import threading
from typing import List, Optional, Coroutine, Any
from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

load_dotenv()
logger = logging.getLogger(__name__)
mcp_port = os.getenv("MCP_SERVER_PORT", "8001")
mcp_host=os.getenv("MCP_SERVER", "0.0.0.0")
MCP_SERVER_URL = f"http://{mcp_host}:{mcp_port}/mcp"

class AsyncEventLoopThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ready = threading.Event()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.ready.set()
        self.loop.run_forever()

    def run_coroutine(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if not self.loop:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result(timeout=30)

_event_loop_thread = AsyncEventLoopThread()
_event_loop_thread.start()
_event_loop_thread.ready.wait()

class MCPClientManager:
    _client: Optional[Client] = None

    @classmethod
    async def ensure_connected(cls):
        if cls._client is None:
            cls._client = Client(MCP_SERVER_URL)
            await cls._client.__aenter__()
        return cls._client

    @classmethod
    async def close(cls):
        if cls._client is not None:
            await cls._client.__aexit__(None, None, None)
            cls._client = None

async def get_mcp_tools_async() -> List[StructuredTool]:
    tools: List[StructuredTool] = []
    client = await MCPClientManager.ensure_connected()
    mcp_tools = await client.list_tools()
    for tool_info in mcp_tools:
        schema = getattr(tool_info, "inputSchema", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        field_definitions = {}
        for param_name, param_schema in properties.items():
            param_type = str
            default_value = param_schema.get("default", None)

            if param_schema.get("type") == "integer":
                param_type = int
            elif param_schema.get("type") == "number":
                param_type = float
            elif param_schema.get("type") == "boolean":
                param_type = bool

            if param_name in required:
                if default_value is not None:
                    field_definitions[param_name] = (param_type, default_value)
                else:
                    field_definitions[param_name] = (param_type, ...)
            else:
                from typing import Optional
                if default_value is not None:
                    field_definitions[param_name] = (param_type, default_value)
                else:
                    field_definitions[param_name] = (Optional[param_type], None)
        
        if field_definitions:
            DynamicModel = create_model(f"{tool_info.name}_Model", **field_definitions)
        else:
            class DynamicModel(BaseModel):
                pass
        
        def create_tool_wrapper(tool_name: str, tool_client=client, model_class=DynamicModel):
            def execute_tool(**kwargs) -> str:
                if set(kwargs.keys()) == {"kwargs"} and isinstance(kwargs["kwargs"], dict):
                    kwargs = kwargs["kwargs"]

                cleaned_kwargs = {}
                for key, value in kwargs.items():
                    if value is None:
                        prop_schema = properties.get(key, {})
                        if "default" in prop_schema:
                            cleaned_kwargs[key] = prop_schema["default"]
                        else:
                            continue
                    else:
                        cleaned_kwargs[key] = value

                async def _execute():
                    result = await tool_client.call_tool(name=tool_name, arguments=cleaned_kwargs)
                    if hasattr(result, "content") and result.content:
                        for content in result.content:
                            if hasattr(content, "text"):
                                return content.text
                    return str(result)

                return _event_loop_thread.run_coroutine(_execute())

            return execute_tool

        func = create_tool_wrapper(tool_info.name)
        tools.append(StructuredTool.from_function(
            func=func,
            name=tool_info.name,
            description=tool_info.description or "",
            args_schema=DynamicModel if field_definitions else None,
            infer_schema=False,
        ))
    return tools


def get_mcp_tools() -> List[StructuredTool]:
    return _event_loop_thread.run_coroutine(get_mcp_tools_async())


def get_tool_by_name(name: str) -> StructuredTool:
    tools = get_mcp_tools()
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool {name} not found in MCP server")
