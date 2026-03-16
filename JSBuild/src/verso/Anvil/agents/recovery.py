import json
import re


class RecoveryManager:
    """
    Manages error recovery and self-healing for agent tasks.
    """

    def __init__(self, master_agent):
        self.master = master_agent

    def handle_failure(self, task, error_msg, attempt_count):
        """
        Analyze failure and propose a correction or retry.
        """
        print(
            f"[!] RECOVERY: Task '{task}' failed (Attempt {attempt_count}). Reason: {error_msg}"
        )

        if attempt_count > 3:
            print("[!] RECOVERY: Max retries exceeded. Aborting task.")
            return None

        # Re-plan or adjust task based on error
        recovery_prompt = f"""
        TASK FAILURE ANALYSIS:
        Task: {task}
        Error: {error_msg}
        Attempt: {attempt_count}
        
        How should we adjust the task or environment to fix this?
        Provide a revised task description or 'RETRY' if it was a transient error.
        """

        adjustment = self.master.brain.generate(recovery_prompt)
        print(f"[*] RECOVERY: Proposed adjustment: {adjustment[:50]}...")

        if "RETRY" in adjustment.upper():
            return task
        else:
            return adjustment

    def sanitize_json(self, raw_text: str) -> dict:
        """
        Extremely robust JSON extraction from LLM responses.
        Handles markdown blocks and stray text.
        """
        # Try to find JSON block
        match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw_text, re.DOTALL)
        if match:
            raw_json = match.group(1)
        else:
            # Fallback to finding first { or [ and last } or ]
            start_idx = min(raw_text.find("{"), raw_text.find("["))
            end_idx = max(raw_text.rfind("}"), raw_text.rfind("]"))
            if start_idx != -1 and end_idx != -1:
                raw_json = raw_text[start_idx : end_idx + 1]
            else:
                raw_json = raw_text

        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            print(f"[!] RECOVERY: Failed to parse JSON from: {raw_text[:100]}...")
            return None
