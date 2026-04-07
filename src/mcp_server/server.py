import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from mcp_server.instance import mcp
import mcp_server.estimation_tool
import mcp_server.geocoding_tool
import mcp_server.tool_bdd

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
