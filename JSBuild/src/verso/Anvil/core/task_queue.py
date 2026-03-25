from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, List
from queue import Queue, Empty
from threading import Thread
import time
import uuid

# Avoid circular imports by importing SubAgent inside methods or using TYPE_CHECKING
# But for type hinting agent_class, we can use Type[Any] or minimal import if possible.


class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedTask:
    task_id: str
    agent_class: Any  # Type[SubAgent]
    description: str
    status: TaskStatus = TaskStatus.QUEUED
    result: Any = None
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    progress: float = 0.0

    @property
    def elapsed_time(self) -> str:
        if self.start_time == 0:
            return "0s"
        end = self.end_time if self.end_time > 0 else time.time()
        return f"{end - self.start_time:.1f}s"


class TaskQueueExecutor:
    """Execute agents sequentially with task queue management"""

    def __init__(self, brain, max_queued: int = 8):
        self.max_queued = max_queued
        self.task_queue: Queue = Queue(maxsize=max_queued)
        self.tasks: Dict[str, QueuedTask] = {}
        self.current_task: Optional[QueuedTask] = None
        self.executor_thread: Optional[Thread] = None
        self.running = False

        # Shared brain instance
        self.shared_brain = brain

    def _generate_task_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def start(self):
        """Start the executor thread"""
        if self.running:
            return

        self.running = True
        self.executor_thread = Thread(target=self._executor_loop, daemon=True)
        self.executor_thread.start()

    def stop(self):
        """Stop the executor thread"""
        self.running = False
        if self.executor_thread:
            self.executor_thread.join(timeout=2.0)

    def submit_task(self, agent_class, task_description: str) -> str:
        """Submit task to queue"""
        if self.task_queue.full():
            raise RuntimeError("Task queue is full (max: 8 tasks)")

        task = QueuedTask(
            task_id=self._generate_task_id(),
            agent_class=agent_class,
            description=task_description,
            status=TaskStatus.QUEUED,
        )

        self.task_queue.put(task)
        self.tasks[task.task_id] = task

        return task.task_id

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task."""
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if task.status != TaskStatus.QUEUED:
            return False  # Can only cancel queued tasks for now

        # Remove from queue? Queue doesn't support random removal.
        # We mark it as cancelled, and the executor loop skips it.
        task.status = TaskStatus.CANCELLED
        return True

    def get_all_tasks(self) -> List[QueuedTask]:
        return list(self.tasks.values())

    def _executor_loop(self):
        """Main executor loop - runs tasks sequentially"""
        while self.running:
            try:
                # Get next task (blocking with timeout)
                task = self.task_queue.get(timeout=1.0)

                # Check cancellation
                if task.status == TaskStatus.CANCELLED:
                    continue

                # Mark as running
                task.status = TaskStatus.RUNNING
                task.start_time = time.time()
                self.current_task = task

                # Execute task with shared brain
                try:
                    result = self._execute_task(task)
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED

                task.end_time = time.time()
                self.current_task = None

                # Mark task done in queue
                self.task_queue.task_done()

            except Empty:
                # No tasks in queue, continue
                continue
            except Exception as e:
                print(f"Executor loop error: {e}")

    def _execute_task(self, task: QueuedTask) -> Any:
        """Execute single task using shared brain"""
        # Create agent instance with shared brain
        # We assume agent_class is a SubAgent subclass
        # We pass a dummy console or handle output capture if needed.
        # For now, let's use a capture console or the main one?
        # Using main console might interfere with REPL typing.
        # Ideally, we capture output.

        from rich.console import Console

        # Using a separate console for capture, or just accept that it prints to stdout
        # But printing to stdout from background thread ruins REPL UX.
        # We should capture output to storing in result if possible.

        # For this implementation, we will pass a Silent Console or Capture Console
        # But SubAgents usually print to self.console.

        # Let's create a Capture Console
        from io import StringIO

        capture_file = StringIO()
        agent_console = Console(file=capture_file, force_terminal=True, width=120)

        agent = task.agent_class(
            task=task.description,
            parent_name="TaskQueue",
            brain=self.shared_brain,  # Reuse model instance
            console=agent_console,
        )

        # Execute
        result = agent.run()

        # If result is just the dict/response, we might want to attach the log
        # Or we rely on result being what it is.
        # For now, return result.

        return result
