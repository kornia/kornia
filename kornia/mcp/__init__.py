"""Model Context Protocol (MCP) server for Kornia.

This module provides MCP server functionality for Kornia's image processing capabilities using the official MCP SDK.
"""
import logging

from .server import create_kornia_mcp_server
from .start import main as cli_main

logger = logging.getLogger(__name__)

__all__ = ['create_kornia_mcp_server', 'cli_main']
