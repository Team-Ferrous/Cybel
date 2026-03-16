import asyncio
import os
import unittest
from unittest.mock import patch, MagicMock
from infrastructure.providers.mcp_provider import MCPProvider


class TestMCPProviderRobustness(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a mock config with a server that requires an env var
        self.mock_config = {
            "mcp_servers": {
                "test-server": {
                    "command": "echo",
                    "args": ["hello"],
                    "env": {"TEST_KEY": "${TEST_KEY}"},
                    "enabled": True,
                }
            }
        }

    @patch("infrastructure.providers.mcp_provider.yaml.safe_load")
    @patch("infrastructure.providers.mcp_provider.os.path.exists")
    @patch("builtins.open")
    async def test_skips_missing_env_var(self, mock_open, mock_exists, mock_yaml):
        mock_exists.return_value = True
        mock_yaml.return_value = self.mock_config

        # Ensure TEST_KEY is not in environment
        if "TEST_KEY" in os.environ:
            del os.environ["TEST_KEY"]

        provider = MCPProvider()

        # This shouldn't raise or hang
        with patch("builtins.print") as mock_print:
            await provider.connect_all()
            mock_print.assert_any_call(
                "Skipping MCP server test-server: Required environment variable TEST_KEY is missing."
            )

        self.assertEqual(len(provider.servers), 0)

    @patch("infrastructure.providers.mcp_provider.yaml.safe_load")
    @patch("infrastructure.providers.mcp_provider.os.path.exists")
    @patch("builtins.open")
    @patch("infrastructure.providers.mcp_provider.stdio_client")
    async def test_timeout_handling(
        self, mock_stdio, mock_open, mock_exists, mock_yaml
    ):
        mock_exists.return_value = True
        mock_yaml.return_value = self.mock_config

        # Set env var so it passes validation
        os.environ["TEST_KEY"] = "dummy"

        # Mock stdio_client to hang
        async def slow_connect(*args, **kwargs):
            await asyncio.sleep(60)  # Longer than the 30s timeout
            return MagicMock()

        mock_stdio.return_value.__aenter__ = slow_connect

        provider = MCPProvider()

        with patch("builtins.print") as mock_print:
            # We lower the timeout for testing if we could,
            # but since it's hardcoded to 30s in the code,
            # we'll just mock the connect_server to raise timeout directly or adjust the test.
            # Actually, let's just test that connect_all catches it.

            with patch(
                "asyncio.timeout", side_effect=lambda t: asyncio.timeout(0.1)
            ):  # Force short timeout
                await provider.connect_all()
                mock_print.assert_any_call(
                    "Skipping MCP server test-server: Connection timed out."
                )


if __name__ == "__main__":
    unittest.main()
