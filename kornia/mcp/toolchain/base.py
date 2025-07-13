from typing import List, Dict, Any
from contextlib import AsyncExitStack
import json
from openai import AsyncOpenAI

from kornia.core.external import mcp as _mcp


class MCPToolChainBase:
    def __init__(
        self,
        session: _mcp.ClientSession,
    ):
        self.session: _mcp.ClientSession = session

    def run_tool_calls(self, tool_name, **kwargs):
        return self.session.call_tool(tool_name, arguments=kwargs)

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format.

        Returns:
            A list of tools in OpenAI format.
        """
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def run_tool_call(self, tool_call):
        result = await self.session.call_tool(
            tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        return result
    

class MCPOpenAIClient(MCPToolChainBase):
    """Client for interacting with OpenAI models using MCP tools."""

    def __init__(
        self,
        api_key: str,
        session: _mcp.ClientSession,
        model: str = "gpt-4o",
    ):
        """Initialize the OpenAI MCP client.

        Args:
            model: The OpenAI model to use.
        """
        # Initialize session and client objects
        super().__init__(session)
        self.exit_stack = AsyncExitStack()
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.model = model
