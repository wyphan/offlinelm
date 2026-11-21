#!/home/wyp/miniconda3/envs/agent/bin/python3
r"""
litert-lm_agent.py

Starts an agent in one-shot or interactive mode for Google Antiravity using LiteRT-LM.
"""

import os
import sys
import asyncio
import argparse

# Ensure this script's directory is always on sys.path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import common as common


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common.add_common_args(parser, include_litertlm_args=True, include_cwd_default=False)
    parser.add_argument(
        "-i", "--initial-prompt",
        help='initial prompt (default in one-shot mode: "Scan <workspaces>, list all files by type, and create a README.md summary.")',
    )
    parser.add_argument(
        "-c", "--chat",
        action="store_true",
        help="run in interactive multi-turn chat mode",
    )
    args = parser.parse_args()

    if args.list:
        common.list_litertlm_models()
        return
        
    if not args.name:
        parser.print_help()
        return

    try:
        model_path = common.resolve_litertlm_model_path(args.name, args.path)
    except FileNotFoundError as e:
        parser.error(str(e))

    _, workspaces = common.resolve_workspaces(args.name, args.workspaces, include_cwd=False)

    policies = common.build_agent_policies(
        workspaces,
        read_only=args.read_only,
        ask_user_handler=common.ask_user_cli_handler,
    )

    config = common.build_local_openai_agent_config(
        args.name,
        base_url=args.base_url,
        read_only=args.read_only,
        workspaces=workspaces,
        policies=policies,
    )

    ws_list_str = ", ".join(workspaces)
    default_prompt = f"Scan {ws_list_str}, list all files by type, and create a README.md summary."
    prompt = None if (args.chat and not args.initial_prompt) else (args.initial_prompt or default_prompt)

    with common.managed_litertlm_server(
        base_url=args.base_url,
        expected_model=args.name,
        device=args.device,
    ):
        asyncio.run(common.run_agent_session(config, prompt, chat_mode=args.chat))


if __name__ == "__main__":
    main()

