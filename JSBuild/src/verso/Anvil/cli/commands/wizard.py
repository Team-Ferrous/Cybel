from typing import List, Optional, Any
from cli.commands.base import SlashCommand
from core.wizard import WorkflowWizard


class WizardCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "wizard"

    @property
    def description(self) -> str:
        return "Launch the Workflow Wizard"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        wiz = WorkflowWizard(context)

        if not args:
            # Interactive mode
            return wiz.run()

        cmd = args[0]

        if cmd == "create":
            if len(args) < 2:
                return "Usage: /wizard create <description>"
            description = " ".join(args[1:])
            return wiz.generate_custom_workflow(description)

        elif cmd == "run":
            # Running named workflows would go here
            return (
                "Usage: /wizard run <name> (Not yet implemented, use interactive mode)"
            )

        return wiz.run()
