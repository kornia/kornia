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

from typing import Any, Dict, List, Optional

from kornia.core import Tensor
from kornia.core.external import mcp as _mcp
from kornia.mcp.toolchain.base import MCPOpenAIClient
from kornia.mcp.utils import base64_to_tensor, tensor_to_base64


class MCPImageProcessingClient(MCPOpenAIClient):
    """Stateful client for single-frame image processing using OpenAI and MCP tools.
    - On first query, asks OpenAI for the toolchain, saves it.
    - Can process new images using the saved toolchain without re-querying OpenAI.
    - Allows updating the toolchain with a new query.
    """

    def __init__(self, api_key: str, session: "_mcp.ClientSession", model: str = "gpt-4o"):
        super().__init__(api_key, session, model)
        self.toolchain: Optional[List[Dict[str, Any]]] = None
        self.message_history: Optional[List[Dict[str, Any]]] = None

    async def process_query(self, query: str, image: Tensor) -> Any:
        """Query OpenAI for a toolchain, process the image, and save the toolchain.
        Returns the final output and saves the toolchain for future use.
        """
        # Query OpenAI and get the toolchain (sequence of tool calls)
        tools = await self.get_mcp_tools()
        if self.message_history is None:
            self.message_history = [
                {
                    "role": "system",
                    "content": "You are an expert assistant. When calling tools, "
                    "always follow the tool documentation and parameter constraints exactly. "
                    "Assume image path is ./input.jpg.",
                }
            ]
        self.message_history.append({"role": "user", "content": query})
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=self.message_history,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        # Save the toolchain (sequence of tool calls)
        self.message_history.append(assistant_message)

        if not assistant_message.tool_calls:
            print("No toolchain/tool calls returned by OpenAI for this query.")
            return
        self.toolchain = [
            {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in assistant_message.tool_calls
        ]
        # Process the image using the toolchain
        return await self.process_image(image)

    async def process_image(self, image: Tensor) -> Any:
        """Process an image using the saved toolchain.
        Returns the final output after applying all tools in the chain.
        """
        if not self.toolchain:
            raise RuntimeError("No toolchain available. Please run process_query first.")
        current_data = tensor_to_base64(image)
        for i, tool in enumerate(self.toolchain):
            # Prepare arguments, assuming 'image' is the main input
            args = dict(**(eval(tool["arguments"]) if isinstance(tool["arguments"], str) else tool["arguments"]))
            for k, v in args.items():
                if k.endswith("_uri"):
                    args[k] = current_data
            # Call the tool via MCP
            result = await self.session.call_tool(tool["name"], arguments=args)
            # Assume the result is the processed image for the next step
            current_data = result.content[0].data if hasattr(result.content[0], "data") else result.content[0].text
            # kornia.io.write_image(f"./int-{i}.jpg", base64_to_tensor(current_data))
        return base64_to_tensor(current_data)

    async def update_toolchain(self, query: str):
        """Update the toolchain state with a new query (does not process an image)."""
        tools = await self.get_mcp_tools()
        messages = [{"role": "user", "content": query}]
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        if not assistant_message.tool_calls:
            raise RuntimeError("No toolchain/tool calls returned by OpenAI for this query.")
        self.toolchain = [
            {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
            for tc in assistant_message.tool_calls
        ]
        self.last_query = query

    async def cleanup(self):
        await self.exit_stack.aclose()
