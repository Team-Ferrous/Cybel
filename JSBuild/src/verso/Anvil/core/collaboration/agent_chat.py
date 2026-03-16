"""Cross-instance master-agent chat channel."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Dict, List, Optional


class AgentChatChannel:
    """Stores and relays chat messages between remote master agents."""

    def __init__(self, transport=None, event_store=None):
        self.transport = transport
        self.event_store = event_store
        self._conversations: Dict[str, List[dict]] = defaultdict(list)

    def _emit(self, event_type: str, payload: dict) -> None:
        if self.event_store is None:
            return
        try:
            self.event_store.emit(event_type=event_type, payload=payload, source="agent_chat")
        except Exception:
            pass

    @staticmethod
    def _run_async(coro) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    def send_message(self, peer_id: str, message: str, context: dict = None):
        payload = {
            "peer_id": peer_id,
            "message": message,
            "context": context or {},
            "timestamp": time.time(),
            "direction": "outbound",
        }
        self._conversations[peer_id].append(payload)
        self._emit("collaboration.chat.sent", payload)

        if self.transport is not None:
            self._run_async(
                self.transport.send(
                    peer_id,
                    {
                        "type": "chat",
                        "sender_id": self.transport.instance.instance_id,
                        "payload": {
                            "message": message,
                            "context": context or {},
                        },
                    },
                )
            )

    def on_message(self, peer_id: str, message: str, context: dict):
        payload = {
            "peer_id": peer_id,
            "message": message,
            "context": context or {},
            "timestamp": time.time(),
            "direction": "inbound",
        }
        self._conversations[peer_id].append(payload)
        self._emit("collaboration.chat.received", payload)

    def get_conversation_log(self, peer_id: str) -> List[dict]:
        return list(self._conversations.get(peer_id, []))
