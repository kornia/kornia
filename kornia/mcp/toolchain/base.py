# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List

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
