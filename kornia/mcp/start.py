"""CLI interface for Kornia MCP server."""

import logging

from .server import create_kornia_mcp_server

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    server = create_kornia_mcp_server()
    server.run()


if __name__ == '__main__':
    main() 