from saguaro.packets.builders import PacketBuilder


class _GraphService:
    def load_graph(self):
        return {"nodes": {}, "relations": {}, "generated_at": 1700000000}


def test_packet_builder_returns_mapping_packet(tmp_path):
    packet = PacketBuilder(str(tmp_path), graph_service=_GraphService()).build_task_packet(
        "bridge templates"
    )

    assert packet["id"].startswith("MAP-")
    assert packet["task"] == "bridge templates"
