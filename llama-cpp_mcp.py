#!/home/wyp/miniconda3/envs/agent/bin/python3
r"""
llama-cpp_mcp.py

Starts a local agent in MCP mode for Google Antigravity using llama.cpp via llama-server.
"""

import os
import sys
import argparse

# Ensure this script's directory is always on sys.path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import common as common
from google.antigravity import Agent
from mcp.server.mcpserver import MCPServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common.add_common_args(parser, include_llamacpp_args=True, include_cwd_default=True)
    args = parser.parse_args()

    if args.list:
        common.list_llamacpp_models(base_url=args.base_url, api_key=args.api_key)
        return

    if not args.name:
        parser.print_help()
        return

    # Only resolve checkpoint & start llama-server AFTER -l and -n checks pass
    try:
        model_path = common.resolve_llamacpp_model_path(args.name, args.path)
    except FileNotFoundError as e:
        parser.error(str(e))

    default_workspace, allowed_workspaces = common.resolve_workspaces(
        args.name, args.workspaces, include_cwd=True
    )

    safe_server_name = args.name.replace("/", "_")
    mcp = MCPServer(f"llamacpp-{safe_server_name}")

    @mcp.tool()
    async def run_llamacpp_agent(
        task: str,
        workspace_dir: str = default_workspace,
    ) -> str:
        """Delegate an autonomous task to the 100% offline llama.cpp agent."""
        target_dir = common.validate_workspace(workspace_dir, allowed_workspaces)
        policies = common.build_agent_policies(
            [target_dir],
            read_only=args.read_only,
            ask_user_handler=None,
        )
        run_config = common.build_local_openai_agent_config(
            args.name,
            base_url=args.base_url,
            api_key=args.api_key,
            read_only=args.read_only,
            workspaces=[target_dir],
            policies=policies,
            enable_tool_logging=False,
        )
        async with Agent(run_config) as agent:
            response = await agent.chat(task)
            return await common.extract_response_text(response)

    common.start_llama_server(
        base_url=args.base_url,
        api_key=args.api_key,
        model_path=model_path,
        model_alias=args.name,
        device=args.device,
        vulkan_device=args.vulkan_device,
    )

    mcp.run()


if __name__ == "__main__":
    main()

