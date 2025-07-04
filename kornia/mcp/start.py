"""CLI interface for Kornia MCP server."""

import logging
import contextlib
from kornia.mcp.server import create_kornia_mcp_server
from kornia.core.external import mcp as _mcp
from kornia.core.external import fastapi

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the CLI."""
    enhance_server = create_kornia_mcp_server("enhance")
    return enhance_server

    # # Create a combined lifespan to manage both session managers
    # @contextlib.asynccontextmanager
    # async def lifespan(app: fastapi.FastAPI):
    #     async with contextlib.AsyncExitStack() as stack:
    #         await stack.enter_async_context(enhance_server.session_manager.run())
    #         yield

    # app = fastapi.FastAPI(lifespan=lifespan)
    # app.mount("/enhance", enhance_server.streamable_http_app())
    # return app

mcp = main()

if __name__ == '__main__':
    mcp.run()
    # import uvicorn
    # uvicorn.run(main(), host="0.0.0.0", port=8000)
