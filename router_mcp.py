"""MCP server exposing the router's tools, so any MCP client (Claude Code, Claude Desktop, the
agent in agent.py with --via-mcp) can use them.

    python router_mcp.py            # stdio transport

Claude Code:  claude mcp add k8s-router -- python /path/to/k8s-issue-router/router_mcp.py
"""
import os

try:  # mcp >= 2 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from router.tools import get_sig_charter, lookup_owners, search_similar_issues  # noqa: E402

mcp = FastMCP("k8s-issue-router")
mcp.tool()(search_similar_issues)
mcp.tool()(get_sig_charter)
mcp.tool()(lookup_owners)

if __name__ == "__main__":
    mcp.run()
