"""SAGUARO MCP Server
Implements Model Context Protocol (MCP) JSON-RPC 2.0 interface.
"""

import json
import logging
import os
import sys
import traceback
from typing import Any

from saguaro.dni.server import DNIServer

# Configure logging
logger = logging.getLogger(__name__)


class MCPServer:
    """Provide MCPServer support."""
    def __init__(self, auth_token: str | None = None) -> None:
        """Initialize the instance."""
        self.dni = DNIServer()
        self.auth_token = auth_token
        self.authenticated = not auth_token

    def run(self) -> None:
        """Main loop for MCP stdio."""
        logger.info("SAGUARO MCP Server Started")
        if self.auth_token:
            logger.info("Authentication enabled.")

        for line in sys.stdin:
            if not line:
                break
            try:
                self.handle_message(line.strip())
            except Exception as e:
                logger.error(f"Error handling message: {e}")

    def handle_message(self, line: str) -> None:
        """Handle handle message."""
        if not line:
            return
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            self.send_error(None, -32700, "Parse error")
            return

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        # Enforce Authentication
        if not self.authenticated and method != "initialize":
            self.send_error(req_id, -32002, "Unauthenticated request.")
            return

        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "notifications/initialized":
                # client acknowledging
                return
            elif method == "tools/list":
                result = self.handle_list_tools()
            elif method == "tools/call":
                result = self.handle_call_tool(params)
            elif method == "resources/list":
                result = self.handle_list_resources()
            elif method == "resources/read":
                result = self.handle_read_resource(params)
            else:
                # Ignore unknown notifications, error on unknown requests
                if req_id is not None:
                    self.send_error(req_id, -32601, f"Method not found: {method}")
                return

            if req_id is not None:
                self.send_result(req_id, result)

        except Exception as e:
            logger.error(f"Error executing {method}: {traceback.format_exc()}")
            if req_id is not None:
                self.send_error(req_id, -32603, f"Internal error: {str(e)}")

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        # Check Authentication
        """Handle handle initialize."""
        if self.auth_token:
            client_token = params.get("authorization")
            # Also support "initializationOptions": { "authorization": "..." }
            if not client_token:
                init_opts = params.get("initializationOptions", {})
                client_token = init_opts.get("authorization")

            if client_token != self.auth_token:
                raise ValueError("Invalid authentication token.")
            self.authenticated = True

        # We can extract rootUri or rootPath from params
        # "capabilities": { "tools": {}, "resources": {} }

        # Look for root path in params
        # Simplified: check for process CWD or manual config
        repo_path = os.getcwd()  # Default

        # Initialize internal DNI with this path
        self.dni.initialize({"path": repo_path})

        return {
            "protocolVersion": "2024-11-05",  # Example MCP version or "0.1.0"
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "SAGUARO", "version": "5.0.0"},
        }

    def handle_list_tools(self) -> dict[str, Any]:
        """Handle handle list tools."""
        return {
            "tools": [
                {
                    "name": "saguaro_query",
                    "description": "Semantic search over the codebase knowledge graph.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "saguaro_read_node",
                    "description": "Read source code for a specific file/node.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file": {
                                "type": "string",
                                "description": "Absolute file path",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Start line (1-indexed)",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "End line (1-indexed)",
                            },
                        },
                        "required": ["file"],
                    },
                },
                {
                    "name": "saguaro_verify",
                    "description": "Verify codebase compliance against rules.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to verify",
                                "default": ".",
                            }
                        },
                    },
                },
            ]
        }

    def handle_call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle handle call tool."""
        name = params.get("name")
        args = params.get("arguments", {})

        if name == "saguaro_query":
            # Map to DNI query
            res = self.dni.query({"text": args.get("query"), "k": args.get("k", 5)})
            # Flatten results for MCP
            content = []
            for r in res.get("results", []):
                score = r["score"]
                path = r["file"]
                line = r["line"]
                # Snippet logic? For now just ref
                content.append(f"[{score:.2f}] {r['name']} ({path}:{line})")

            return {"content": [{"type": "text", "text": "\n".join(content)}]}

        if name == "saguaro_read_node":
            res = self.dni.read_node(args)
            return {"content": [{"type": "text", "text": res.get("content", "")}]}

        if name == "saguaro_verify":
            # Call Verifier (need to expose via DNI or import directly)
            # For now, import directly as DNI doesn't have verify yet
            from saguaro.sentinel.verifier import SentinelVerifier

            verifier = SentinelVerifier(
                repo_path=os.path.abspath(args.get("path", "."))
            )
            violations = verifier.verify_all()

            if violations:
                text = f"Found {len(violations)} violations:\n"
                for v in violations:
                    text += (
                        f"- [{v['severity']}] {v['file']}:{v['line']} {v['message']}\n"
                    )
            else:
                text = "No violations found."

            return {"content": [{"type": "text", "text": text}]}

        raise ValueError(f"Unknown tool: {name}")

    def handle_list_resources(self) -> dict[str, Any]:
        # Expose some basic resources?
        # Maybe the .saguaro/config.yaml
        """Handle handle list resources."""
        return {
            "resources": [
                {
                    "uri": "saguaro://stats",
                    "name": "Index Statistics",
                    "mimeType": "application/json",
                }
            ]
        }

    def handle_read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle handle read resource."""
        uri = params.get("uri")
        if uri == "saguaro://stats":
            stats = self.dni.status({})
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(stats, indent=2),
                    }
                ]
            }
        raise ValueError(f"Resource not found: {uri}")

    def send_result(self, req_id: Any, result: Any) -> None:
        """Handle send result."""
        res = {"jsonrpc": "2.0", "result": result, "id": req_id}
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

    def send_error(self, req_id: Any, code: int, message: str) -> None:
        """Handle send error."""
        res = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()


def main() -> None:
    # Parse generic args or environment for auth
    """Handle main."""
    auth_token = os.environ.get("saguaro_MCP_TOKEN")
    # Simple arg parsing manually to avoid conflict with mcp stdio if any
    for i, arg in enumerate(sys.argv):
        if arg == "--auth-token" and i + 1 < len(sys.argv):
            auth_token = sys.argv[i + 1]

    server = MCPServer(auth_token=auth_token)
    server.run()


if __name__ == "__main__":
    main()
