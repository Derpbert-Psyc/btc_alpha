"""
Phase 5 — CLI Skeleton (7 subcommands via argparse)

Commands: status, run, halt, resume, replay, config, inspect
All output as JSON envelope.

References:
    PHASE5_INTEGRATION_SEAMS_v1_2_6.md:365-430 — CLI contract
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from phase5_bundle_store import RunMode
from phase5_pod import (
    HealthState,
    Pod,
    PodConfig,
    PodState,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON envelope helper
# ---------------------------------------------------------------------------

def _envelope(
    pod_id: str,
    command: str,
    run_mode: str,
    health: str,
    result: Any,
) -> str:
    """Format standard JSON output envelope."""
    d = {
        "pod_id": pod_id,
        "command": command,
        "timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z",
        "run_mode": run_mode,
        "health": health,
        "result": result,
    }
    return json.dumps(d, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# Pod registry (in-process — for CLI demo / testing)
# ---------------------------------------------------------------------------

_POD_REGISTRY: Dict[str, Pod] = {}


def get_or_create_pod(
    pod_id: str,
    run_mode: str = "RESEARCH",
    evidence_dir: str = "/tmp/btc_alpha_evidence",
) -> Pod:
    """Get existing pod or create one."""
    if pod_id in _POD_REGISTRY:
        return _POD_REGISTRY[pod_id]

    config = PodConfig(
        pod_id=pod_id,
        stream_id="BTCUSDT-1m",
        run_mode=run_mode,
        starting_capital_cents=1_000_000,
        friction_preset="CONSERVATIVE",
        bar_interval_seconds=60,
        evidence_dir=evidence_dir,
    )
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    pod = Pod(config, run_id)
    _POD_REGISTRY[pod_id] = pod
    return pod


def get_pod(pod_id: str) -> Optional[Pod]:
    """Get existing pod or None."""
    return _POD_REGISTRY.get(pod_id)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> str:
    """status --pod-id <id>"""
    pod = get_pod(args.pod_id)
    if pod is None:
        return _envelope(args.pod_id, "status", "UNKNOWN", "UNKNOWN",
                         {"error": f"Pod {args.pod_id} not found"})

    st = pod.status()
    return _envelope(
        pod.pod_id, "status", pod.run_mode.value,
        pod.overall_health().value, st,
    )


def cmd_run(args: argparse.Namespace) -> str:
    """run --pod-id <id> --mode RESEARCH|SHADOW|LIVE [--data <path>]"""
    evidence_dir = getattr(args, "evidence_dir", "/tmp/btc_alpha_evidence")
    pod = get_or_create_pod(args.pod_id, args.mode, evidence_dir)
    return _envelope(
        pod.pod_id, "run", pod.run_mode.value,
        pod.overall_health().value,
        {"message": f"Pod {pod.pod_id} running in {pod.run_mode.value} mode"},
    )


def cmd_halt(args: argparse.Namespace) -> str:
    """halt --pod-id <id> --reason <text>"""
    pod = get_pod(args.pod_id)
    if pod is None:
        return _envelope(args.pod_id, "halt", "UNKNOWN", "UNKNOWN",
                         {"error": f"Pod {args.pod_id} not found"})

    pod.halt(args.reason)
    return _envelope(
        pod.pod_id, "halt", pod.run_mode.value,
        pod.overall_health().value,
        {"message": f"Pod {pod.pod_id} HALTED: {args.reason}"},
    )


def cmd_resume(args: argparse.Namespace) -> str:
    """resume --pod-id <id>  (interactive confirmation)"""
    pod = get_pod(args.pod_id)
    if pod is None:
        return _envelope(args.pod_id, "resume", "UNKNOWN", "UNKNOWN",
                         {"error": f"Pod {args.pod_id} not found"})

    # Interactive confirmation: user must type the pod_id
    if sys.stdin.isatty():
        print(f"To resume pod, type the pod ID ({pod.pod_id}): ", end="", flush=True)
        confirmation = input().strip()
    else:
        confirmation = args.pod_id  # Non-interactive: use provided pod_id

    success = pod.resume(confirmation)
    return _envelope(
        pod.pod_id, "resume", pod.run_mode.value,
        pod.overall_health().value,
        {"success": success, "message": "Resumed" if success else "Resume failed"},
    )


def cmd_replay(args: argparse.Namespace) -> str:
    """replay --pod-id <id> --from-bar <n>"""
    pod = get_pod(args.pod_id)
    if pod is None:
        return _envelope(args.pod_id, "replay", "UNKNOWN", "UNKNOWN",
                         {"error": f"Pod {args.pod_id} not found"})

    store = pod.get_bundle_store()
    from_seq = args.from_bar
    to_seq = store.watermark.commit_seq

    if from_seq > to_seq:
        return _envelope(
            pod.pod_id, "replay", pod.run_mode.value,
            pod.overall_health().value,
            {"error": f"from_bar {from_seq} > watermark {to_seq}"},
        )

    bundles = store.replay(from_seq, to_seq)
    summary = [
        {
            "commit_seq": b.commit_seq,
            "type": "bar" if hasattr(b, "bar_index") else "event",
        }
        for b in bundles
    ]
    return _envelope(
        pod.pod_id, "replay", pod.run_mode.value,
        pod.overall_health().value,
        {"bundles_count": len(bundles), "summary": summary},
    )


def cmd_config(args: argparse.Namespace) -> str:
    """config --pod-id <id>"""
    pod = get_pod(args.pod_id)
    if pod is None:
        return _envelope(args.pod_id, "config", "UNKNOWN", "UNKNOWN",
                         {"error": f"Pod {args.pod_id} not found"})

    from dataclasses import asdict
    return _envelope(
        pod.pod_id, "config", pod.run_mode.value,
        pod.overall_health().value,
        asdict(pod.config),
    )


def cmd_inspect(args: argparse.Namespace) -> str:
    """inspect --pod-id <id>  (FORBIDDEN in LIVE)"""
    pod = get_pod(args.pod_id)
    if pod is None:
        return _envelope(args.pod_id, "inspect", "UNKNOWN", "UNKNOWN",
                         {"error": f"Pod {args.pod_id} not found"})

    if pod.run_mode == RunMode.LIVE:
        return _envelope(
            pod.pod_id, "inspect", pod.run_mode.value,
            pod.overall_health().value,
            {"error": "inspect is FORBIDDEN in LIVE mode"},
        )

    st = pod.status()
    # Add store details
    store = pod.get_bundle_store()
    st["store_dir"] = store.store_dir
    chain_ok, chain_err = store.verify_chain_integrity()
    st["chain_integrity"] = {"valid": chain_ok, "error": chain_err}

    return _envelope(
        pod.pod_id, "inspect", pod.run_mode.value,
        pod.overall_health().value,
        st,
    )


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build argparse parser with 7 subcommands."""
    parser = argparse.ArgumentParser(
        prog="phase5_cli",
        description="BTC Alpha Phase 5 CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Pod health, mode, state")
    p_status.add_argument("--pod-id", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Begin computation")
    p_run.add_argument("--pod-id", required=True)
    p_run.add_argument("--mode", required=True, choices=["RESEARCH", "SHADOW", "LIVE"])
    p_run.add_argument("--data", default=None, help="Data path")
    p_run.add_argument("--evidence-dir", default="/tmp/btc_alpha_evidence")

    # halt
    p_halt = subparsers.add_parser("halt", help="Immediate HALT")
    p_halt.add_argument("--pod-id", required=True)
    p_halt.add_argument("--reason", required=True)

    # resume
    p_resume = subparsers.add_parser("resume", help="Resume from HALTED")
    p_resume.add_argument("--pod-id", required=True)

    # replay
    p_replay = subparsers.add_parser("replay", help="Deterministic replay")
    p_replay.add_argument("--pod-id", required=True)
    p_replay.add_argument("--from-bar", type=int, required=True)

    # config
    p_config = subparsers.add_parser("config", help="Display immutable config")
    p_config.add_argument("--pod-id", required=True)

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Internal state (FORBIDDEN in LIVE)")
    p_inspect.add_argument("--pod-id", required=True)

    return parser


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

COMMAND_HANDLERS = {
    "status": cmd_status,
    "run": cmd_run,
    "halt": cmd_halt,
    "resume": cmd_resume,
    "replay": cmd_replay,
    "config": cmd_config,
    "inspect": cmd_inspect,
}


def main(argv: Optional[list] = None) -> str:
    """Parse args and dispatch to handler.  Returns JSON output."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMAND_HANDLERS[args.command]
    return handler(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    output = main()
    print(output)
