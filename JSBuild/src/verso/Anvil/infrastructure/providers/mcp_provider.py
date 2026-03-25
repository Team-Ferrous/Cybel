import asyncio
import os
import yaml
from typing import Dict, List, Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - optional dependency in test envs
    ClientSession = Any
    StdioServerParameters = None
    stdio_client = None


class MCPProvider:
    """
    Provider for Model Context Protocol (MCP) servers.
    Standardizes tool interfaces across different external services.
    """

    def __init__(self, config_path: str = "config/mcp_servers.yaml"):
        self.config_path = config_path
        self.servers: Dict[str, ClientSession] = {}
        self.server_contexts = {}  # Store contexts to prevent GC
        self.configs = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f).get("mcp_servers", {})

    async def connect_all(self):
        """Connect to all enabled servers in config."""
        for name, config in self.configs.items():
            if config.get("enabled", False):
                try:
                    await self.connect_server(name)
                except ValueError as ve:
                    print(f"Skipping MCP server {name}: {ve}")
                except asyncio.TimeoutError:
                    print(f"Skipping MCP server {name}: Connection timed out.")
                except Exception as e:
                    print(f"Failed to connect to MCP server {name}: {e}")

    async def connect_server(self, name: str):
        """Connect to a specific MCP server by name."""
        if StdioServerParameters is None or stdio_client is None:
            raise RuntimeError("MCP dependencies are not installed.")
        if name not in self.configs:
            raise ValueError(f"Server {name} not found in config.")

        config = self.configs[name]
        env = config.get("env", {})
        # Resolve environment variables
        resolved_env = {}
        for k, v in env.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                var_name = v[2:-1]
                val = os.getenv(var_name)
                if val is None:
                    raise ValueError(
                        f"Required environment variable {var_name} is missing."
                    )
                resolved_env[k] = val
            else:
                resolved_env[k] = v

        server_params = StdioServerParameters(
            command=config["command"], args=config["args"], env=resolved_env
        )

        # We need to keep the context alive
        try:
            async with asyncio.timeout(30):  # 30 second timeout for connection
                ctx = stdio_client(server_params)
                read, write = await ctx.__aenter__()
                session = ClientSession(read, write)
                await session.initialize()

                self.servers[name] = session
                self.server_contexts[name] = ctx

                return await session.list_tools()
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Connection to MCP server {name} timed out.")

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Execute a tool on an MCP server."""
        if server_name not in self.servers:
            await self.connect_server(server_name)

        session = self.servers[server_name]
        result = await session.call_tool(tool_name, arguments=arguments)
        return result

    async def list_all_tools(self) -> Dict[str, List[Any]]:
        """Return a map of server names to their available tools."""
        all_tools = {}
        for name, session in self.servers.items():
            tools = await session.list_tools()
            all_tools[name] = tools
        return all_tools

    async def disconnect_all(self):
        """Cleanly shutdown all MCP connections."""
        for name, ctx in self.server_contexts.items():
            await ctx.__aexit__(None, None, None)
        self.servers.clear()
        self.server_contexts.clear()


# Singleton instance
_mcp_provider = None


def get_mcp_provider() -> MCPProvider:
    global _mcp_provider
    if _mcp_provider is None:
        _mcp_provider = MCPProvider()
    return _mcp_provider
