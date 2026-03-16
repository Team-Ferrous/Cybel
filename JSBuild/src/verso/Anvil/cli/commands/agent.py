import os
from typing import Any, List

from cli.commands.base import SlashCommand
from config.settings import CAMPAIGN_CONFIG


class AgentCommand(SlashCommand):
    name = "agent"
    description = "Start an autonomous agentic loop"

    def execute(self, args: List[str], context: Any) -> str:
        if not args:
            return "Usage: /agent <objective>"

        objective = " ".join(args)
        context.console.print(
            f"[bold cyan]Starting Agent Mission:[/bold cyan] {objective}"
        )
        context.run_mission(objective)
        return None  # Loop handles output


class CampaignCommand(SlashCommand):
    name = "campaign"
    description = "Create, compile, run, and inspect campaign orchestrations"

    def execute(self, args: List[str], context: Any) -> str:
        if not CAMPAIGN_CONFIG.get("enabled", True):
            return "Campaign orchestrator is disabled in settings."

        if not args:
            return (
                "Usage: /campaign [create|compile|run|list|status|resume|ledger|attach-repo|acquire-repos|questionnaire|feature-map|roadmap|artifacts|approve|continue|detach|timeline|dashboard|speculate|promote-branch|audit|adopt-rule|rule-outcome|closure-proof] ..."
            )

        from core.campaign import CampaignGenerator, CampaignRunner

        action = args[0].lower()
        runner = CampaignRunner(
            brain_factory=lambda: getattr(context, "brain", None),
            console=getattr(context, "console", None),
            ownership_registry=getattr(context, "ownership_registry", None),
        )
        generator = CampaignGenerator(agent=context)

        if action == "create":
            if len(args) < 2:
                return "Usage: /campaign create <description>"
            description = " ".join(args[1:])
            managed = runner.create_autonomy_campaign(
                name=description[:64],
                objective=description,
                directives=[description],
                root_dir=".",
            )
            context.current_campaign_id = managed["campaign_id"]
            path = generator.generate_from_description(description, target_repo=".")
            return (
                f"Managed campaign: {managed['campaign_id']}\n"
                f"Generated campaign: {path}"
            )

        if action == "compile":
            if len(args) < 2:
                return "Usage: /campaign compile <spec.yaml>"
            yaml_path = args[1]
            if not os.path.exists(yaml_path):
                return f"YAML spec not found: {yaml_path}"
            path = generator.generate_from_yaml(yaml_path)
            return f"Compiled campaign: {path}"

        if action == "run":
            if len(args) < 2:
                return "Usage: /campaign run <name-or-path>"
            report = runner.run_campaign(args[1], root_dir=".")
            completed = sum(
                1 for status in report.phase_statuses.values() if status == "passed"
            )
            return (
                f"Campaign completed: {report.campaign_id} "
                f"({completed}/{len(report.phase_statuses)} phases passed)"
            )

        if action == "list":
            campaigns = runner.list_campaigns()
            if not campaigns:
                return "No campaigns found."
            return "\n".join(
                [
                    f"{item['name']} ({item['category']}) -> {item['path']}"
                    for item in campaigns
                ]
            )

        if action == "status":
            if len(args) < 2:
                return "Usage: /campaign status <campaign_id>"
            status = runner.get_campaign_status(args[1])
            if status is None:
                return f"No campaign state found: {args[1]}"
            context.current_campaign_id = args[1]
            phases = status.get("phase_statuses", {})
            control_plane = status.get("control_plane") or {}
            snapshot = status.get("snapshot") or {}
            return (
                f"Campaign: {status.get('campaign_name')}\n"
                f"Started: {status.get('started_at')}\n"
                f"Completed: {status.get('completed_at')}\n"
                f"State: {control_plane.get('current_state')}\n"
                f"Repo dossier: {(snapshot.get('repo_dossier_brief') or {}).get('brief_summary', 'n/a')}\n"
                f"Risk: {(snapshot.get('roadmap_risk_summary') or {}).get('top_item', 'n/a')} / {(snapshot.get('roadmap_risk_summary') or {}).get('high_risk_count', 0)} high\n"
                f"Detached: {(status.get('detached') or {}).get('state', 'n/a')}\n"
                f"Phases: {phases}"
            )

        if action == "resume":
            if len(args) < 2:
                return "Usage: /campaign resume <campaign_id>"
            report = runner.resume_campaign(args[1], root_dir=".")
            context.current_campaign_id = args[1]
            return f"Campaign resumed and completed: {report.campaign_id}"

        if action == "ledger":
            if len(args) < 2:
                return "Usage: /campaign ledger <campaign_id>"
            summary = runner.get_ledger_summary(args[1], budget_tokens=1500)
            return summary

        if action == "attach-repo":
            if len(args) < 4:
                return "Usage: /campaign attach-repo <campaign_id> <role> <path>"
            result = runner.attach_repo(args[1], role=args[2], repo_path=args[3])
            return f"Attached repo {result['repo_id']} as {result['role']}"

        if action == "acquire-repos":
            if len(args) < 3:
                return "Usage: /campaign acquire-repos <campaign_id> <path-or-url> [path-or-url ...]"
            result = runner.acquire_repos(args[1], repo_specs=args[2:])
            return f"Acquired {result['count']} repos into analysis storage."

        if action == "questionnaire":
            if len(args) < 2:
                return "Usage: /campaign questionnaire <campaign_id>"
            result = runner.build_questionnaire(args[1])
            return f"Questionnaire built with {result['count']} questions."

        if action == "feature-map":
            if len(args) < 2:
                return "Usage: /campaign feature-map <campaign_id>"
            result = runner.build_feature_map(args[1])
            return result["rendered"]

        if action == "roadmap":
            if len(args) < 2:
                return "Usage: /campaign roadmap <campaign_id>"
            result = runner.build_roadmap(args[1])
            return (
                f"Roadmap items: {len(result['items'])}\n"
                f"Validation errors: {result['validation_errors']}"
            )

        if action == "artifacts":
            if len(args) < 2:
                return "Usage: /campaign artifacts <campaign_id>"
            artifacts = runner.list_artifacts(args[1])
            if not artifacts:
                return "No campaign artifacts found."
            return "\n".join(
                [
                    f"{item['artifact_id']} [{item['family']}] approval={item['approval_state']} blocking={item['blocking']}"
                    for item in artifacts
                ]
            )

        if action == "approve":
            if len(args) < 3:
                return "Usage: /campaign approve <campaign_id> <artifact_id>"
            runner.approve_artifact(args[1], args[2])
            return f"Approved artifact: {args[2]}"

        if action == "continue":
            if len(args) < 2:
                return "Usage: /campaign continue <campaign_id>"
            event = runner.continue_autonomy_campaign(args[1])
            context.current_campaign_id = args[1]
            return f"Campaign transitioned to {event['to_state']}"

        if action == "detach":
            if len(args) < 2:
                return "Usage: /campaign detach <campaign_id>"
            status = runner.detach_campaign(args[1])
            context.current_campaign_id = args[1]
            return (
                f"Detached campaign worker launched for {args[1]} "
                f"(pid={status.get('pid')}, state={status.get('state')})"
            )

        if action == "timeline":
            if len(args) < 2:
                return "Usage: /campaign timeline <campaign_id>"
            payload = runner.timeline(args[1])
            return (
                f"Timeline: {payload['path']}\n"
                f"Transitions: {payload['summary']['transition_count']}\n"
                f"Telemetry: {payload['summary']['telemetry_count']}"
            )

        if action == "dashboard":
            if len(args) < 2:
                return "Usage: /campaign dashboard <campaign_id>"
            payload = runner.specialist_dashboard(args[1])
            return (
                f"Specialist dashboard packets={payload['summary']['packet_count']} "
                f"accepted={payload['summary']['accepted_count']} "
                f"failed={payload['summary']['failed_count']}"
            )

        if action == "speculate":
            if len(args) < 3:
                return "Usage: /campaign speculate <campaign_id> <roadmap_item_id>"
            payload = runner.speculate_roadmap_item(args[1], args[2])
            return (
                f"Speculation: {payload['comparison_id']}\n"
                f"Winner: {payload['winner_lane_id']}\n"
                f"Ghost verifier: {'present' if payload.get('ghost_verifier') else 'skipped'}"
            )

        if action == "promote-branch":
            if len(args) < 4:
                return "Usage: /campaign promote-branch <campaign_id> <comparison_id> <branch_lane_id>"
            payload = runner.promote_speculative_branch(args[1], args[2], args[3])
            return (
                f"Promoted {args[3]} -> "
                f"{len(payload['promoted']['promoted_files'])} files"
            )

        if action == "audit":
            if len(args) < 2:
                return "Usage: /campaign audit <campaign_id>"
            result = runner.run_audit(args[1])
            return f"Audit findings: {result['summary']['finding_count']}"

        if action == "adopt-rule":
            if len(args) < 3:
                return "Usage: /campaign adopt-rule <campaign_id> <rule_id> [notes]"
            notes = " ".join(args[3:]) if len(args) > 3 else ""
            payload = runner.adopt_rule_proposal(args[1], args[2], notes=notes)
            return f"Adopted rule: {payload['rule_id']}"

        if action == "rule-outcome":
            if len(args) < 4:
                return "Usage: /campaign rule-outcome <campaign_id> <rule_id> <outcome_status> [notes]"
            notes = " ".join(args[4:]) if len(args) > 4 else ""
            payload = runner.record_rule_outcome(
                args[1],
                args[2],
                outcome_status=args[3],
                notes=notes,
            )
            return f"Rule outcome recorded: {payload['rule_id']} -> {payload['outcome_status']}"

        if action == "closure-proof":
            if len(args) < 2:
                return "Usage: /campaign closure-proof <campaign_id>"
            result = runner.closure_proof(args[1])
            return (
                f"Closure proof: {result['path']}\n"
                f"Closure allowed: {result['proof']['closure_allowed']}"
            )

        return "Usage: /campaign [create|compile|run|list|status|resume|ledger|attach-repo|acquire-repos|questionnaire|feature-map|roadmap|artifacts|approve|continue|detach|timeline|dashboard|speculate|promote-branch|audit|adopt-rule|rule-outcome|closure-proof]"


class ThinkingCommand(SlashCommand):
    name = "thinking"
    description = "Toggle extended thinking visibility"

    def execute(self, args: List[str], context: Any) -> str:
        if not args:
            state = "ON" if getattr(context, "show_thinking", True) else "OFF"
            return f"Thinking is currently {state}. Usage: /thinking [on|off]"

        mode = args[0].lower()
        if mode == "on":
            context.show_thinking = True
            return "Extended thinking display ENABLED."
        elif mode == "off":
            context.show_thinking = False
            return "Extended thinking display DISABLED."
        else:
            return "Usage: /thinking [on|off]"


class OwnershipCommand(SlashCommand):
    name = "ownership"
    description = "Manage file ownership state"

    def execute(self, args: List[str], context: Any) -> str:
        registry = getattr(context, "ownership_registry", None)
        if registry is None:
            return "Ownership is disabled or not initialized."

        if not args or args[0] == "status":
            snapshot = registry.get_status_snapshot()
            return (
                f"Claimed files: {snapshot.get('total_claimed_files', 0)} | "
                f"Agents: {snapshot.get('total_agents', 0)}"
            )

        action = args[0].lower()
        if action == "query":
            if len(args) < 2:
                return "Usage: /ownership query <file1> [file2 ...]"
            records = registry.query_ownership(args[1:])
            if not records:
                return "No ownership records found for the requested files."
            lines = []
            for file_path, record in records.items():
                lines.append(
                    f"{file_path}: {record.owner_agent_id} ({record.mode}, instance={record.owner_instance_id})"
                )
            return "\n".join(lines)

        if action == "release-all":
            registry.release_all()
            return "Released all ownership leases."

        return "Usage: /ownership [status|query|release-all]"


class PeersCommand(SlashCommand):
    name = "peers"
    description = "Inspect discovered collaboration peers"

    def execute(self, args: List[str], context: Any) -> str:
        discovery = getattr(context, "peer_discovery", None)
        repo_presence = getattr(context, "repo_presence", None)
        if discovery is None:
            return (
                "Peer discovery unavailable "
                "(collaboration may be running in local-only mode)."
            )

        action = args[0].lower() if args else "list"
        if action != "list":
            return "Usage: /peers list"

        if repo_presence is not None:
            snapshot = repo_presence.refresh()
            peers = snapshot.get("peers", [])
        else:
            discovery.refresh()
            peers = discovery.get_peers(same_project_only=True)
        if not peers:
            return "No peers discovered for this project."
        return "\n".join(
            [
                (
                    f"{peer['instance_id']} @ {peer['listen_address']} "
                    f"phase={peer.get('phase_id') or '-'} "
                    f"campaign={peer.get('campaign_id') or '-'} "
                    f"claims={peer.get('active_claim_count', 0)} "
                    f"provider={peer.get('transport_provider', 'unknown')} "
                    f"trust={peer.get('trust_zone', 'internal')} "
                    f"verified={peer.get('verification_state', 'unknown')} "
                    f"connected={peer.get('connected', False)}"
                )
                if isinstance(peer, dict)
                else f"{peer.instance_id} @ {peer.listen_address} ({peer.hostname}/{peer.user})"
                for peer in peers
            ]
        )


class CollaborateCommand(SlashCommand):
    name = "collaborate"
    description = "Manage cross-instance collaboration mode"

    def execute(self, args: List[str], context: Any) -> str:
        if not args:
            enabled = bool(getattr(context, "collaboration_enabled", False))
            mode = str(getattr(context, "collaboration_mode", "disabled"))
            return (
                f"Collaboration is {'ENABLED' if enabled else 'DISABLED'} "
                f"(mode={mode})."
            )

        action = args[0].lower()
        if action == "enable":
            context.collaboration_enabled = True
            peer_discovery = getattr(context, "peer_discovery", None)
            if peer_discovery is not None:
                peer_discovery.start()
                context.collaboration_mode = "networked"
                return "Collaboration ENABLED (networked)."
            context.collaboration_mode = "local"
            return "Collaboration ENABLED (local-only; no peer transport)."

        if action == "disable":
            context.collaboration_enabled = False
            if getattr(context, "peer_discovery", None) is not None:
                context.peer_discovery.stop()
            context.collaboration_mode = "disabled"
            return "Collaboration DISABLED."

        if action == "proposals":
            negotiator = getattr(context, "collaboration_negotiator", None)
            proposals = (
                list(getattr(negotiator, "proposals", {}).values())
                if negotiator
                else []
            )
            if not proposals:
                return "No collaboration proposals."
            return "\n".join(
                [
                    f"{proposal.get('proposal_id')}: status={proposal.get('status', 'proposed')}"
                    for proposal in proposals[-10:]
                ]
            )

        if action == "chat-log":
            chat_channel = getattr(context, "agent_chat_channel", None)
            if chat_channel is None:
                return "Agent chat channel is not initialized."
            if len(args) < 2:
                return "Usage: /collaborate chat-log <peer_id>"
            peer_id = args[1]
            log = chat_channel.get_conversation_log(peer_id)
            if not log:
                return f"No chat history with peer '{peer_id}'."
            lines = []
            for entry in log[-20:]:
                direction = entry.get("direction", "?")
                lines.append(f"[{direction}] {entry.get('message', '')}")
            return "\n".join(lines)

        available = ", ".join(["enable", "disable", "proposals", "chat-log"])
        return f"Usage: /collaborate [{available}]"
