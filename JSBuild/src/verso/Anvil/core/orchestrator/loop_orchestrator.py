from core.agent import BaseAgent
from core.context import ContextManager
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
from cli.renderer import CLIRenderer
from core.unified_chat_loop import UnifiedChatLoop
from cli.progress_dashboard import LiveProgressDashboard


class LoopOrchestrator:
    def __init__(
        self,
        agent: BaseAgent,
        saguaro: SaguaroSubstrate,
        token_manager: ContextManager,
        renderer: CLIRenderer,
    ):
        self.agent = agent
        self.saguaro = saguaro
        self.token_manager = token_manager
        self.renderer = renderer
        self.dashboard = None

        # Initialize the powerful UnifiedChatLoop
        # The 'enhanced_loop_enabled' flag is read from the agent (REPL instance)
        self.unified_loop = UnifiedChatLoop(
            agent=self.agent,
            enhanced_mode=getattr(self.agent, "enhanced_loop_enabled", True),
        )

    @property
    def thinking_system(self):
        """Expose the thinking system from the unified loop."""
        return self.unified_loop.thinking_system

    def run(self, objective: str):
        """
        Runs the mission by delegating to the UnifiedChatLoop.
        Creates a live progress dashboard for real-time visibility.
        """
        # Create dashboard for this mission
        self.dashboard = LiveProgressDashboard(
            console=self.renderer.console,
            title="Mission Execution",
            renderer=self.renderer,
        )

        # Add UPORE phases for visibility (Understand → Plan → Execute → Observe → Repeat)
        self.dashboard.add_phase("Understand", "pending", "Classifying request...")
        self.dashboard.add_phase("Plan", "pending", "Gathering evidence & strategy...")
        self.dashboard.add_phase("Execute", "pending", "Processing actions...")
        self.dashboard.add_phase("Observe", "pending", "Verifying results...")
        self.dashboard.add_phase("Synthesize", "pending", "Generating response...")

        # Run the unified loop with dashboard for real-time updates
        return self.unified_loop.run(objective, dashboard=self.dashboard)
