from cli.commands.agent import CollaborateCommand, PeersCommand


class _Context:
    def __init__(self):
        self.collaboration_enabled = False
        self.collaboration_mode = "disabled"
        self.peer_discovery = None


def test_collaborate_enable_without_peer_transport_uses_local_mode():
    context = _Context()
    command = CollaborateCommand()

    result = command.execute(["enable"], context)

    assert context.collaboration_enabled is True
    assert context.collaboration_mode == "local"
    assert "local-only" in result


def test_peers_command_reports_local_only_when_discovery_unavailable():
    context = _Context()
    command = PeersCommand()

    result = command.execute(["list"], context)

    assert "local-only" in result
