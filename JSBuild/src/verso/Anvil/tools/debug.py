import subprocess


def run_with_debugger(command: str) -> str:
    """
    Runs a command and if it fails, attempts to provide a 'debug' context.
    For Python, it can attempt to run with a hook to capture locals.
    """
    # Simple implementation: run the command and capture output
    # If it fails, report the error prominently.

    try:
        # If it's a python command, we could inject a debugging script
        # but for now let's just run and capture.
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120
        )

        output = result.stdout
        error = result.stderr

        if result.returncode == 0:
            return f"Command execution successful.\nOutput:\n{output}"
        else:
            # Try to provide more context for "Self-Healing"
            analysis = "\n[DEBUG ANALYSIS]\n"
            if "Traceback" in error:
                analysis += "Detected Python traceback. Identifying failing line...\n"
                # Extract the last few lines of the traceback
                tb_lines = error.splitlines()
                for line in reversed(tb_lines):
                    if line.startswith("  File"):
                        analysis += f"Probable failure point: {line.strip()}\n"
                        break

            diagnosis = self_heal_diagnose(f"{error}\n{output}")
            return (
                f"Command FAILED (Exit Code: {result.returncode})\n\n"
                f"[STDOUT]\n{output}\n\n"
                f"[STDERR]\n{error}\n"
                f"{analysis}\n"
                f"[SELF-HEAL]\n{diagnosis}"
            )

    except Exception as e:
        return f"Error running debugger: {str(e)}"


def self_heal_diagnose(error_report: str) -> str:
    """
    Analyzes an error report and suggests a fix (intended for agent to call).
    """
    lowered = (error_report or "").lower()

    if "modulenotfounderror" in lowered or "no module named" in lowered:
        return "Missing Python dependency detected. Install the missing package in the active environment and rerun."
    if "filenotfounderror" in lowered or "no such file or directory" in lowered:
        return "Missing file/path detected. Verify working directory and input paths before retrying."
    if "command not found" in lowered:
        return "Missing system binary detected. Install the referenced command or adjust PATH."
    if "permission denied" in lowered:
        return "Permission issue detected. Check file permissions and execution rights."
    if "syntaxerror" in lowered:
        return "Syntax error detected. Fix the reported line/column and rerun."
    if "timed out" in lowered or "timeout" in lowered:
        return "Execution timed out. Reduce scope or increase timeout and retry."

    return "Review stderr and traceback for the first failing frame, patch that root cause, then rerun."
