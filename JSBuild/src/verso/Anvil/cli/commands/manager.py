from typing import Any, List, Optional

from cli.commands.base import SlashCommand


class ManagerCommand(SlashCommand):
    """Detached campaign manager and log inspector."""

    @property
    def name(self) -> str:
        return "manager"

    @property
    def description(self) -> str:
        return "Inspect detached campaigns, timelines, speculation lanes, and worker logs"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.campaign.runner import CampaignRunner

        runner = CampaignRunner(
            brain_factory=lambda: getattr(context, "brain", None),
            console=getattr(context, "console", None),
            config=getattr(context, "campaign_runner_config", None),
            ownership_registry=getattr(context, "ownership_registry", None),
        )
        action = args[0].lower() if args else "campaigns"

        if action == "campaigns":
            statuses = runner.list_detached_campaigns()
            if not statuses:
                return "No detached campaigns found."
            return "\n".join(
                [
                    f"{item['campaign_id']}: {item.get('state', 'unknown')} "
                    f"log={item.get('log_path', 'n/a')} "
                    f"lane={((item.get('detached_lane') or {}).get('workspace_dir', 'n/a'))}"
                    for item in statuses
                ]
            )

        if action == "status":
            if len(args) < 2:
                return "Usage: /manager status <campaign_id>"
            detached = runner.detached_status(args[1])
            campaign = runner.get_campaign_status(args[1]) or {}
            risk = ((campaign.get("snapshot") or {}).get("roadmap_risk_summary") or {})
            return str(
                {
                    "detached": detached,
                    "roadmap_risk_summary": risk,
                    "current_state": ((campaign.get("control_plane") or {}).get("current_state")),
                }
            )

        if action == "watch":
            if len(args) < 2:
                return "Usage: /manager watch <campaign_id>"
            log_text = runner.detached_log(args[1], lines=30)
            timeline = runner.timeline(args[1])
            return (
                (log_text or "No detached log output.")
                + "\n---\n"
                + f"Timeline: {timeline.get('path')}\n"
                + f"Transitions: {timeline.get('summary', {}).get('transition_count', 0)}"
            )

        if action == "cancel":
            if len(args) < 2:
                return "Usage: /manager cancel <campaign_id>"
            status = runner.cancel_detached(args[1])
            return f"{args[1]} -> {status.get('state', 'unknown')}"

        if action == "speculate":
            if len(args) < 3:
                return "Usage: /manager speculate <campaign_id> <roadmap_item_id>"
            payload = runner.speculate_roadmap_item(args[1], args[2])
            return (
                f"{payload['comparison_id']} winner={payload['winner_lane_id']} "
                f"ghost={'yes' if payload.get('ghost_verifier') else 'no'}"
            )

        if action == "promote":
            if len(args) < 4:
                return "Usage: /manager promote <campaign_id> <comparison_id> <branch_lane_id>"
            payload = runner.promote_speculative_branch(args[1], args[2], args[3])
            return (
                f"promoted={len(payload['promoted']['promoted_files'])} "
                f"branch={args[3]}"
            )

        return "Usage: /manager [campaigns|status|watch|cancel|speculate|promote] ..."
