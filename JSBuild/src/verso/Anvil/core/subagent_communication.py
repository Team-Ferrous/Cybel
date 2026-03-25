"""
Subagent Communication Protocols for Anvil

Phase 4.2: Message passing, coordination, and shared context management
for multi-agent workflows.
"""

import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from threading import RLock
from rich.console import Console
from rich.table import Table
from rich import box

from shared_kernel.event_store import get_event_store


class MessageType(Enum):
    """Types of messages exchanged between subagents."""

    REQUEST = "request"  # Request for work/data
    RESPONSE = "response"  # Response to request
    HANDOFF = "handoff"  # Transfer control to another agent
    BROADCAST = "broadcast"  # Send to all agents
    STATUS_UPDATE = "status"  # Status notification
    ERROR = "error"  # Error notification
    RESULT = "result"  # Task result
    COORDINATION = "coordination"  # Coordination signal
    OWNERSHIP_CLAIMED = "ownership_claimed"
    OWNERSHIP_RELEASED = "ownership_released"
    OWNERSHIP_DENIED = "ownership_denied"
    OWNERSHIP_QUERY = "ownership_query"
    OWNERSHIP_TRANSFER = "ownership_transfer"
    PHASE_ASSIGNED = "phase_assigned"
    PHASE_COMPLETED = "phase_completed"


OWNERSHIP_TOPICS = [
    "ownership.claims",
    "ownership.conflicts",
    "ownership.phases",
    "ownership.heartbeats",
]


class Priority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    """
    Message exchanged between subagents.

    Attributes:
        message_id: Unique identifier
        sender: Agent ID of sender
        recipient: Agent ID of recipient (None for broadcast)
        message_type: Type of message
        payload: Message content/data
        priority: Message priority
        timestamp: Creation timestamp
        reply_to: ID of message being replied to
        metadata: Additional metadata
    """

    message_id: str
    sender: str
    recipient: Optional[str]
    message_type: MessageType
    payload: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentChannel:
    """Communication channel for a subagent."""

    agent_id: str
    inbox: deque = field(default_factory=deque)
    outbox: deque = field(default_factory=deque)
    subscriptions: List[str] = field(default_factory=list)  # Event subscriptions
    status: str = "active"  # active, busy, idle, stopped
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """
    Central message bus for inter-agent communication.

    Phase 4.2: Implements pub/sub, point-to-point, and broadcast patterns.
    Thread-safe for parallel agent execution.
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.event_store = get_event_store()

        # Agent channels
        self.channels: Dict[str, AgentChannel] = {}

        # Message routing
        self.message_queue: deque = deque()
        self.message_history: List[Message] = []

        # Event subscriptions (topic -> [agent_ids])
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)

        # Coordination state
        self.shared_context: Dict[str, Any] = {}

        # Thread safety
        self.lock = RLock()

        # Message handlers (topic -> callback)
        self.handlers: Dict[str, Callable] = {}
        self.trace_segments: List[Dict[str, Any]] = []
        self.trace_sink: Optional[Callable[[Dict[str, Any]], None]] = None
        self.trace_context: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            "total_messages": 0,
            "messages_by_type": defaultdict(int),
            "messages_by_agent": defaultdict(int),
        }

    def register_agent(
        self,
        agent_id: str,
        subscriptions: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> AgentChannel:
        """
        Register a subagent with the message bus.

        Args:
            agent_id: Unique agent identifier
            subscriptions: List of topics to subscribe to
            metadata: Additional agent metadata

        Returns:
            AgentChannel for the agent
        """
        with self.lock:
            if agent_id in self.channels:
                self.console.print(
                    f"[yellow]⚠ Agent {agent_id} already registered, reusing channel[/yellow]"
                )
                return self.channels[agent_id]

            requested_subscriptions = list(subscriptions or [])
            for topic in OWNERSHIP_TOPICS:
                if topic not in requested_subscriptions:
                    requested_subscriptions.append(topic)

            channel = AgentChannel(
                agent_id=agent_id,
                subscriptions=requested_subscriptions,
                metadata=metadata or {},
            )

            self.channels[agent_id] = channel

            # Register subscriptions
            for topic in channel.subscriptions:
                self.subscriptions[topic].append(agent_id)

            self.console.print(f"[green]✓ Registered agent: {agent_id}[/green]")
            return channel

    def unregister_agent(self, agent_id: str):
        """Unregister an agent and clean up its channel."""
        with self.lock:
            if agent_id not in self.channels:
                return

            channel = self.channels[agent_id]

            # Remove subscriptions
            for topic in channel.subscriptions:
                if agent_id in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(agent_id)

            # Clear channel
            del self.channels[agent_id]

            self.console.print(f"[dim]Agent {agent_id} unregistered[/dim]")

    def send(
        self,
        sender: str,
        recipient: Optional[str],
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Send a message from one agent to another (or broadcast).

        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID (None for broadcast)
            message_type: Type of message
            payload: Message content
            priority: Message priority
            reply_to: Message ID being replied to
            metadata: Additional metadata

        Returns:
            Message ID
        """
        trace_segment = None
        with self.lock:
            # Generate message ID
            message_id = f"msg_{int(time.time() * 1000000)}"

            # Create message
            message = Message(
                message_id=message_id,
                sender=sender,
                recipient=recipient,
                message_type=message_type,
                payload=payload,
                priority=priority,
                reply_to=reply_to,
                metadata=metadata or {},
            )

            # Route message
            if recipient is None:
                # Broadcast to all agents
                self._broadcast_message(message)
            else:
                # Point-to-point delivery
                self._deliver_message(message)

            # Store in history
            self.message_history.append(message)

            # Update stats
            self.stats["total_messages"] += 1
            self.stats["messages_by_type"][message_type.value] += 1
            self.stats["messages_by_agent"][sender] += 1
            trace_segment = self._capture_trace_segment(
                action="send",
                message=message,
            )

        self._emit_trace_segment(trace_segment)
        return message_id

    def _deliver_message(self, message: Message):
        """Deliver message to specific recipient."""
        recipient = message.recipient

        if recipient not in self.channels:
            self.console.print(f"[red]✗ Recipient {recipient} not found[/red]")
            return

        channel = self.channels[recipient]

        # Add to inbox (priority queue)
        if message.priority == Priority.CRITICAL:
            channel.inbox.appendleft(message)  # Front of queue
        else:
            channel.inbox.append(message)  # Back of queue

        self.console.print(
            f"[dim]→ Message delivered: {message.sender} → {recipient} ({message.message_type.value})[/dim]"
        )

    def _broadcast_message(self, message: Message):
        """Broadcast message to all registered agents."""
        for agent_id, channel in self.channels.items():
            if agent_id != message.sender:  # Don't send to self
                channel.inbox.append(message)

        self.console.print(
            f"[dim]→ Broadcast from {message.sender} to {len(self.channels) - 1} agents[/dim]"
        )

    def publish(
        self,
        topic: str,
        sender: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Publish an event to all subscribers of a topic.

        Args:
            topic: Event topic
            sender: Publisher agent ID
            payload: Event payload
            priority: Message priority
            metadata: Additional metadata

        Returns:
            Message ID
        """
        handler = None
        handler_message = None
        trace_segment = None
        with self.lock:
            message_id = f"event_{int(time.time() * 1000000)}"

            # Create event message
            message = Message(
                message_id=message_id,
                sender=sender,
                recipient=None,  # Will be delivered to subscribers
                message_type=MessageType.BROADCAST,
                payload=payload,
                priority=priority,
                metadata={"topic": topic, **(metadata or {})},
            )

            # Deliver to all subscribers
            subscribers = self.subscriptions.get(topic, [])

            for subscriber_id in subscribers:
                if subscriber_id != sender and subscriber_id in self.channels:
                    channel = self.channels[subscriber_id]
                    channel.inbox.append(message)

            self.message_history.append(message)
            self.stats["total_messages"] += 1
            self.stats["messages_by_type"][MessageType.BROADCAST.value] += 1
            self.stats["messages_by_agent"][sender] += 1

            self.console.print(
                f"[dim]→ Event published: {topic} to {len(subscribers)} subscribers[/dim]"
            )

            handler = self.handlers.get(topic)
            handler_message = message
            trace_segment = self._capture_trace_segment(
                action="publish",
                message=message,
                extra={
                    "topic": topic,
                    "subscriber_count": len(subscribers),
                },
            )

        self._emit_trace_segment(trace_segment)
        if handler:
            try:
                handler(handler_message)
            except Exception as exc:
                self.console.print(
                    f"[yellow]⚠ Topic handler error for '{topic}': {exc}[/yellow]"
                )

        return message_id

    def register_handler(self, topic: str, callback: Callable[[Message], None]) -> None:
        """Register a direct callback for topic events."""
        with self.lock:
            self.handlers[topic] = callback

    def receive(
        self, agent_id: str, timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Receive next message from agent's inbox.

        Args:
            agent_id: Agent receiving the message
            timeout: Optional timeout in seconds

        Returns:
            Message or None if inbox empty
        """
        with self.lock:
            if agent_id not in self.channels:
                return None

            channel = self.channels[agent_id]

            if channel.inbox:
                message = channel.inbox.popleft()
                trace_segment = self._capture_trace_segment(
                    action="receive",
                    message=message,
                    extra={"receiver": agent_id},
                )
                self._emit_trace_segment(trace_segment)
                return message

            return None

    def peek(self, agent_id: str) -> Optional[Message]:
        """Peek at next message without removing it."""
        with self.lock:
            if agent_id not in self.channels:
                return None

            channel = self.channels[agent_id]

            if channel.inbox:
                return channel.inbox[0]

            return None

    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to a topic."""
        with self.lock:
            if agent_id not in self.channels:
                self.console.print(f"[red]✗ Agent {agent_id} not registered[/red]")
                return

            if topic not in self.channels[agent_id].subscriptions:
                self.channels[agent_id].subscriptions.append(topic)
                self.subscriptions[topic].append(agent_id)

                self.console.print(f"[dim]Agent {agent_id} subscribed to {topic}[/dim]")

    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe agent from a topic."""
        with self.lock:
            if agent_id not in self.channels:
                return

            if topic in self.channels[agent_id].subscriptions:
                self.channels[agent_id].subscriptions.remove(topic)

                if agent_id in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(agent_id)

                self.console.print(
                    f"[dim]Agent {agent_id} unsubscribed from {topic}[/dim]"
                )

    def update_status(self, agent_id: str, status: str):
        """Update agent status."""
        with self.lock:
            if agent_id in self.channels:
                self.channels[agent_id].status = status

    def get_shared_context(self, key: str) -> Any:
        """Get value from shared context."""
        with self.lock:
            return self.shared_context.get(key)

    def set_shared_context(self, key: str, value: Any):
        """Set value in shared context."""
        with self.lock:
            self.shared_context[key] = value

        # Notify subscribers of context change outside the lock.
        self.publish(
            topic=f"context.{key}",
            sender="system",
            payload={"key": key, "value": value},
        )

    def clear_shared_context(self):
        """Clear all shared context."""
        with self.lock:
            self.shared_context.clear()

    def get_channel_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for an agent's channel."""
        with self.lock:
            if agent_id not in self.channels:
                return {}

            channel = self.channels[agent_id]

            return {
                "agent_id": agent_id,
                "inbox_size": len(channel.inbox),
                "outbox_size": len(channel.outbox),
                "subscriptions": channel.subscriptions,
                "status": channel.status,
                "metadata": channel.metadata,
            }

    def get_bus_stats(self) -> Dict[str, Any]:
        """Get overall message bus statistics."""
        with self.lock:
            return {
                "total_agents": len(self.channels),
                "total_messages": self.stats["total_messages"],
                "messages_by_type": dict(self.stats["messages_by_type"]),
                "messages_by_agent": dict(self.stats["messages_by_agent"]),
                "active_subscriptions": {
                    topic: len(subs) for topic, subs in self.subscriptions.items()
                },
                "message_history_size": len(self.message_history),
            }

    def print_stats(self):
        """Print message bus statistics as a Rich table."""
        stats = self.get_bus_stats()

        self.console.print("\n[bold magenta]📡 Message Bus Statistics[/bold magenta]\n")

        # Overview table
        overview = Table(show_header=False, box=box.ROUNDED)
        overview.add_column("Metric", style="cyan")
        overview.add_column("Value", style="white")

        overview.add_row("Total Agents", str(stats["total_agents"]))
        overview.add_row("Total Messages", str(stats["total_messages"]))
        overview.add_row("Message History", str(stats["message_history_size"]))

        self.console.print(overview)

        # Messages by type
        if stats["messages_by_type"]:
            self.console.print("\n[bold]Messages by Type:[/bold]\n")

            type_table = Table(
                show_header=True, header_style="bold cyan", box=box.SIMPLE
            )
            type_table.add_column("Type", style="green")
            type_table.add_column("Count", style="yellow", justify="right")

            for msg_type, count in sorted(
                stats["messages_by_type"].items(), key=lambda x: x[1], reverse=True
            ):
                type_table.add_row(msg_type, str(count))

            self.console.print(type_table)

        # Active agents
        if self.channels:
            self.console.print("\n[bold]Active Agents:[/bold]\n")

            agent_table = Table(
                show_header=True, header_style="bold cyan", box=box.SIMPLE
            )
            agent_table.add_column("Agent ID", style="green")
            agent_table.add_column("Status", style="cyan")
            agent_table.add_column("Inbox", style="yellow", justify="right")
            agent_table.add_column("Subscriptions", style="dim")

            for agent_id, channel in self.channels.items():
                agent_table.add_row(
                    agent_id,
                    channel.status,
                    str(len(channel.inbox)),
                    ", ".join(channel.subscriptions[:3])
                    + ("..." if len(channel.subscriptions) > 3 else ""),
                )

            self.console.print(agent_table)

    def visualize_message_flow(self):
        """Visualize recent message flow."""
        from rich.tree import Tree

        self.console.print("\n[bold magenta]📨 Recent Message Flow[/bold magenta]\n")

        # Show last 20 messages
        recent = self.message_history[-20:]

        if not recent:
            self.console.print("[dim]No messages yet[/dim]")
            return

        tree = Tree("[bold]Message Flow[/bold]")

        for msg in recent:
            recipient_str = msg.recipient or "ALL"

            msg_node = tree.add(
                f"[cyan]{msg.sender}[/cyan] → [green]{recipient_str}[/green] "
                f"({msg.message_type.value}, priority: {msg.priority.value})"
            )

            # Show payload summary
            payload_str = str(msg.payload)[:50]
            if len(str(msg.payload)) > 50:
                payload_str += "..."

            msg_node.add(f"[dim]Payload: {payload_str}[/dim]")

            if msg.reply_to:
                msg_node.add(f"[dim]Reply to: {msg.reply_to}[/dim]")

        self.console.print(tree)

    def export_message_log(self, output_path: str) -> str:
        """Export message history to JSON file."""
        export_data = {
            "exported_at": time.time(),
            "total_messages": len(self.message_history),
            "stats": self.get_bus_stats(),
            "trace_context": dict(self.trace_context),
            "trace_segments": list(self.trace_segments),
            "messages": [
                {
                    "message_id": msg.message_id,
                    "sender": msg.sender,
                    "recipient": msg.recipient,
                    "message_type": msg.message_type.value,
                    "priority": msg.priority.value,
                    "timestamp": msg.timestamp,
                    "payload": msg.payload,
                    "reply_to": msg.reply_to,
                    "metadata": msg.metadata,
                }
                for msg in self.message_history
            ],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        self.console.print(f"[green]✔ Message log exported to {output_path}[/green]")
        return output_path

    def export_trace_segments(self, output_path: str) -> str:
        """Export normalized trace segments to JSON file."""
        export_data = {
            "exported_at": time.time(),
            "trace_context": dict(self.trace_context),
            "total_segments": len(self.trace_segments),
            "segments": list(self.trace_segments),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        self.console.print(f"[green]✔ Message trace exported to {output_path}[/green]")
        return output_path

    def attach_trace_sink(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Attach a callback that receives normalized trace segments."""
        with self.lock:
            self.trace_sink = callback

    def set_trace_context(
        self,
        *,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        """Set ambient trace context carried on future segments."""
        with self.lock:
            if run_id is not None:
                self.trace_context["run_id"] = run_id
            if task_id is not None:
                self.trace_context["task_id"] = task_id
            if phase is not None:
                self.trace_context["phase"] = phase

    def _capture_trace_segment(
        self,
        *,
        action: str,
        message: Message,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        segment = {
            "segment_id": f"seg_{int(time.time() * 1000000)}",
            "action": action,
            "run_id": self.trace_context.get("run_id"),
            "task_id": self.trace_context.get("task_id"),
            "phase": self.trace_context.get("phase"),
            "message_id": message.message_id,
            "timestamp": message.timestamp,
            "sender": message.sender,
            "recipient": message.recipient,
            "message_type": message.message_type.value,
            "priority": message.priority.value,
            "reply_to": message.reply_to,
            "metadata": dict(message.metadata),
            "payload": dict(message.payload),
        }
        if extra:
            segment.update(dict(extra))
        self.trace_segments.append(segment)
        return segment

    def _emit_trace_segment(self, segment: Optional[Dict[str, Any]]) -> None:
        if not segment:
            return
        try:
            self.event_store.emit(
                event_type="TRACE_SEGMENT",
                source=str(segment.get("sender") or "message_bus"),
                payload={
                    "run_id": segment.get("run_id"),
                    "segment_id": segment.get("segment_id"),
                    "message_id": segment.get("message_id"),
                    "message_type": segment.get("message_type"),
                    "action": segment.get("action"),
                    "topic": segment.get("topic"),
                    "files": list((segment.get("payload") or {}).get("files") or []),
                    "symbols": list(
                        (segment.get("payload") or {}).get("symbols") or []
                    ),
                    "tool_calls": list(
                        (segment.get("payload") or {}).get("tool_calls") or []
                    ),
                },
                metadata={
                    "run_id": segment.get("run_id"),
                    "phase": segment.get("phase"),
                    "files": list((segment.get("payload") or {}).get("files") or []),
                    "symbols": list(
                        (segment.get("payload") or {}).get("symbols") or []
                    ),
                    "tool_calls": list(
                        (segment.get("payload") or {}).get("tool_calls") or []
                    ),
                    "segment_id": segment.get("segment_id"),
                },
                run_id=segment.get("run_id"),
            )
        except Exception:
            pass
        sink = self.trace_sink
        if sink is None:
            return
        try:
            sink(dict(segment))
        except Exception as exc:
            self.console.print(
                f"[yellow]⚠ Trace sink error: {exc}[/yellow]"
            )


class CoordinationProtocol:
    """
    High-level coordination protocols for multi-agent workflows.

    Phase 4.2: Implements common coordination patterns.
    """

    def __init__(self, message_bus: MessageBus, console: Optional[Console] = None):
        self.bus = message_bus
        self.console = console or Console()

    def handoff(
        self,
        from_agent: str,
        to_agent: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Handoff task from one agent to another.

        Args:
            from_agent: Agent handing off
            to_agent: Agent receiving handoff
            payload: Task data and context
            metadata: Additional metadata

        Returns:
            Message ID
        """
        self.console.print(f"[cyan]🔄 Handoff: {from_agent} → {to_agent}[/cyan]")

        return self.bus.send(
            sender=from_agent,
            recipient=to_agent,
            message_type=MessageType.HANDOFF,
            payload=payload,
            priority=Priority.HIGH,
            metadata=metadata,
        )

    def barrier_sync(self, agent_id: str, barrier_id: str, total_agents: int) -> bool:
        """
        Barrier synchronization: wait for all agents to reach a checkpoint.

        Args:
            agent_id: Agent reaching barrier
            barrier_id: Unique barrier identifier
            total_agents: Total number of agents expected

        Returns:
            True if all agents have reached the barrier
        """
        # Use shared context to track barrier state
        barrier_key = f"barrier_{barrier_id}"
        barrier_state = self.bus.get_shared_context(barrier_key) or {
            "count": 0,
            "agents": [],
        }

        if agent_id not in barrier_state["agents"]:
            barrier_state["agents"].append(agent_id)
            barrier_state["count"] += 1
            self.bus.set_shared_context(barrier_key, barrier_state)

        if barrier_state["count"] >= total_agents:
            self.console.print(
                f"[green]✓ Barrier {barrier_id} reached by all {total_agents} agents[/green]"
            )

            # Broadcast completion
            self.bus.publish(
                topic=f"barrier.{barrier_id}.complete",
                sender="coordinator",
                payload={"barrier_id": barrier_id, "agents": barrier_state["agents"]},
            )

            return True
        else:
            self.console.print(
                f"[dim]Barrier {barrier_id}: {barrier_state['count']}/{total_agents} agents[/dim]"
            )
            return False

    def request_response(
        self,
        requester: str,
        responder: str,
        request_payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Request-response pattern: send request and wait for response.

        Args:
            requester: Agent making request
            responder: Agent handling request
            request_payload: Request data
            timeout: Response timeout in seconds

        Returns:
            Response payload or None if timeout
        """
        # Send request
        request_id = self.bus.send(
            sender=requester,
            recipient=responder,
            message_type=MessageType.REQUEST,
            payload=request_payload,
            priority=Priority.HIGH,
        )

        # Wait for response
        start_time = time.time()

        while time.time() - start_time < timeout:
            msg = self.bus.receive(requester)

            if (
                msg
                and msg.message_type == MessageType.RESPONSE
                and msg.reply_to == request_id
            ):
                return msg.payload

            time.sleep(0.1)

        self.console.print(
            f"[red]✗ Request-response timeout: {requester} → {responder}[/red]"
        )
        return None

    def aggregate_results(
        self,
        coordinator: str,
        worker_agents: List[str],
        task_topic: str,
        timeout: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """
        Scatter-gather pattern: coordinate parallel tasks and aggregate results.

        Args:
            coordinator: Coordinating agent
            worker_agents: List of worker agent IDs
            task_topic: Topic for task distribution
            timeout: Aggregation timeout

        Returns:
            List of results from workers
        """
        # Subscribe coordinator to result topic
        result_topic = f"{task_topic}.result"
        self.bus.subscribe(coordinator, result_topic)

        results = []
        received_from = set()
        start_time = time.time()

        # Wait for all results
        while (
            len(received_from) < len(worker_agents)
            and time.time() - start_time < timeout
        ):
            msg = self.bus.receive(coordinator)

            if msg and msg.metadata.get("topic") == result_topic:
                results.append(msg.payload)
                received_from.add(msg.sender)

        if len(received_from) < len(worker_agents):
            self.console.print(
                f"[yellow]⚠ Aggregation incomplete: received {len(received_from)}/{len(worker_agents)} results[/yellow]"
            )
        else:
            self.console.print(
                f"[green]✓ Aggregated {len(results)} results from workers[/green]"
            )

        return results


# Global message bus instance
_global_message_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create global message bus instance."""
    global _global_message_bus

    if _global_message_bus is None:
        _global_message_bus = MessageBus()

    return _global_message_bus
