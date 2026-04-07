import asyncio
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from mcp_server.instance import mcp

async def test_tools():
    print("--- Testing MCP Tools ---")
    print("\nTesting geocoding_search('Tours')...")
    try:
        res = await mcp.call_tool("geocoding_search", {"q": "Tours", "limit": 1})
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error calling geocoding_search: {e}")

    print("\nTesting estimate_property(commune='Blois', type_bien='Maison', surface=100)...")
    try:
        res = await mcp.call_tool("estimate_property", {
            "commune": "Blois",
            "type_bien": "Maison",
            "surface": 100,
            "rooms": 4
        })
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error calling estimate_property: {e}")

    print("\nTesting get_database_schema()...")
    try:
        res = await mcp.call_tool("get_database_schema", {})
        print(f"Result: {res}")
    except Exception as e:
        print(f"Error calling get_database_schema: {e}")

if __name__ == "__main__":
    asyncio.run(test_tools())