import unittest
import time
from unittest.mock import MagicMock
from core.task_queue import TaskQueueExecutor, TaskStatus


# Mock SubAgent
class MockSubAgent:
    def __init__(self, task, parent_name, brain, console=None):
        self.task = task
        self.brain = brain
        self.console = console

    def run(self):
        time.sleep(0.1)  # Simulate work
        return "Task Completed"


class TestTaskQueue(unittest.TestCase):
    def setUp(self):
        self.mock_brain = MagicMock()
        self.executor = TaskQueueExecutor(self.mock_brain, max_queued=5)
        self.executor.start()

    def tearDown(self):
        self.executor.stop()

    def test_submission_and_execution(self):
        # Submit task
        task_id = self.executor.submit_task(MockSubAgent, "Test Task")
        self.assertIsNotNone(task_id)

        # Check queued
        tasks = self.executor.get_all_tasks()
        self.assertEqual(len(tasks), 1)
        # It might pick it up immediately

        # Wait for completion
        time.sleep(0.5)

        updated_task = self.executor.tasks[task_id]
        self.assertEqual(updated_task.status, TaskStatus.COMPLETED)
        self.assertEqual(updated_task.result, "Task Completed")

    def test_cancellation(self):
        # Create a slow task so we can queue another behind it
        class SlowAgent(MockSubAgent):
            def run(self):
                time.sleep(1.0)
                return "Slow Done"

        self.executor.submit_task(SlowAgent, "Slow Task")
        t2 = self.executor.submit_task(MockSubAgent, "Waiting Task")

        # Cancel t2 while t1 is running
        cancelled = self.executor.cancel_task(t2)
        self.assertTrue(cancelled)

        task2 = self.executor.tasks[t2]
        self.assertEqual(task2.status, TaskStatus.CANCELLED)

        # Wait for t1 to finish
        time.sleep(1.5)

        # t2 should not run, so result should be None
        self.assertIsNone(task2.result)


if __name__ == "__main__":
    unittest.main()
