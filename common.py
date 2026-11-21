#!/home/wyp/miniconda3/envs/agent/bin/python3
r"""
common.py

Shared helpers for local LiteRT-LM and llama.cpp agents (CLI and MCP).
"""

import os
import time
import shutil
import atexit
import subprocess
import asyncio
import argparse
import json
import urllib.request
from urllib.parse import urlparse
from contextlib import contextmanager

# Vulkan ICD must be set before importing litert_lm or google.antigravity
if "VK_DRIVER_FILES" not in os.environ and "VK_ICD_FILENAMES" not in os.environ:
    vulkan_icd_dirs = (
        "/etc/vulkan/icd.d",
        "/usr/share/vulkan/icd.d",
        "/usr/local/share/vulkan/icd.d",
    )
    vulkan_icd_candidates = (
        # NVIDIA
        "nvidia_icd.x86_64.json",
        "nvidia_icd.json",
        # AMD (prefer Mesa RADV over AMDVLK)
        "radeon_icd.x86_64.json",
        "radeon_icd.json",
        "amd_icd64.json",
        # Intel
        "intel_icd.x86_64.json",
        "intel_icd.json",
        "intel_hasvk_icd.x86_64.json",
        "intel_hasvk_icd.json",
    )
    for icd_dir in vulkan_icd_dirs:
        found = next(
            (
                os.path.join(icd_dir, fname)
                for fname in vulkan_icd_candidates
                if os.path.isfile(os.path.join(icd_dir, fname))
            ),
            None,
        )
        if found:
            os.environ["VK_DRIVER_FILES"] = found
            break

# Suppress unverified device warning on AMD APUs
os.environ.setdefault("ANTIGRAVITY_ALLOW_CPU", "1")

from google.antigravity import Agent, hooks, LocalOpenAIAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.hooks.hooks import HookResult
from google.antigravity.hooks.policy import AskUserHandler
from google.antigravity.types import CapabilitiesConfig, BuiltinTools, CompactionConfig, RetryConfig, ModelOutputRetryConfig

litertlm_server_host_default = "127.0.0.1"
litertlm_server_port_default = "9379"
litertlm_server_url_default = f"http://{litertlm_server_host_default}:{litertlm_server_port_default}/v1"

llama_server_host_default = "127.0.0.1"
llama_server_port_default = "8000"
llama_server_url_default = f"http://{llama_server_host_default}:{llama_server_port_default}/v1"


def add_common_args(
    parser: argparse.ArgumentParser,
    include_cwd_default: bool = False,
    include_litertlm_args: bool = False,
    include_llamacpp_args: bool = False,
) -> None:
    """Register CLI flags shared by offline agent and MCP scripts."""
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="list available models",
    )
    parser.add_argument(
        "-n", "--name",
        help="model name",
    )
    cwd_note = "'.' and " if include_cwd_default else ""
    parser.add_argument(
        "-w", "--workspaces",
        nargs="+",
        help=f"additional allowed agent workspaces (default always includes {cwd_note}'~/.agents/<model_name>/my_workspace')",
    )
    parser.add_argument(
        "-r", "--read-only",
        action="store_true",
        help="restrict agent to read-only built-in tools (disables file edits, creation, and shell commands)",
    )
    parser.add_argument(
        "-d", "--device",
        choices=["cpu", "vulkan", "cuda"],
        default="vulkan" if include_llamacpp_args else None,
        help="compute backend device",
    )

    if include_litertlm_args:
        parser.add_argument(
            "-p", "--path",
            help="path to .litertlm model checkpoint file",
        )
        parser.add_argument(
            "-u", "--base-url",
            default=litertlm_server_url_default,
            help=f"OpenAI-compatible base URL for litert-lm serve (default: {litertlm_server_url_default})",
        )

    if include_llamacpp_args:
        parser.add_argument(
            "-p", "--path",
            help="path to .gguf model checkpoint file",
        )
        parser.add_argument(
            "-u", "--base-url",
            default=llama_server_url_default,
            help=f"OpenAI-compatible base URL for llama-server (default: {llama_server_url_default})",
        )
        parser.add_argument(
            "-k", "--api-key",
            default="local",
            help="API key for llama-server if --api-key was configured (default: local)",
        )
        parser.add_argument(
            "--gpu-device",
            default=None,
            help="specific GPU device ID for llama-server (e.g. CUDA0, Vulkan0; default: auto-select)",
        )


def list_litertlm_models(
        models_root: str = "~/.litert-lm/models",
) -> None:
    """Print available model directories under ~/.litert-lm/models."""
    root = os.path.expanduser(models_root)
    if not os.path.isdir(root):
        print(f"No models directory found at {models_root}")
        return
    models = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    print("Available models:")
    for d in models:
        print("  -", d)


def resolve_litertlm_model_path(
        model_name: str,
        custom_path: str | None = None,
) -> str:
    """Resolve and validate the .litertlm checkpoint path."""
    default_path = os.path.expanduser(f"~/.litert-lm/models/{model_name}/model.litertlm")
    resolved = os.path.realpath(os.path.expanduser(custom_path) if custom_path else default_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
    return resolved


def _discover_cached_ggufs(cache_dir: str | None = None) -> list[tuple[str, str]]:
    """
    Return sorted (display_name, real_path) pairs for cached .gguf weights
    across ~/.cache/huggingface/hub and ~/.cache/llama.cpp (or LLAMA_CACHE).
    """
    roots = [
        os.path.expanduser(cache_dir or os.environ.get("LLAMA_CACHE", "~/.cache/llama.cpp")),
        os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")),
    ]
    seen_real: set[str] = set()
    results: list[tuple[str, str]] = []

    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if not fname.endswith(".gguf") or "mmproj" in fname.lower():
                    continue
                full_path = os.path.join(dirpath, fname)
                real_path = os.path.realpath(full_path)
                if not os.path.isfile(real_path) or real_path in seen_real:
                    continue
                seen_real.add(real_path)

                # Extract "org/repo" if inside HF Hub's models--org--repo/snapshots/... layout
                rel_parts = os.path.relpath(dirpath, root).split(os.sep)
                repo_prefix = ""
                if rel_parts and rel_parts[0].startswith("models--"):
                    repo_prefix = rel_parts[0][len("models--"):].replace("--", "/") + " -> "
                results.append((f"{repo_prefix}{fname}", real_path))

    return sorted(results, key=lambda item: item[0].lower())


def list_llamacpp_models(
    base_url: str = llama_server_url_default,
    api_key: str = "local",
    cache_dir: str | None = None,
) -> None:
    """List cached .gguf checkpoints (HF Hub + llama.cpp cache) and any models on a running llama-server."""
    cached = _discover_cached_ggufs(cache_dir)
    print("Cached GGUF models:")
    if cached:
        for label, path in cached:
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(f"  - {label} ({size_gb:.2f} GB)")
    else:
        print("  (none found in ~/.cache/huggingface/hub or ~/.cache/llama.cpp)")

    if _is_openai_server_ready(base_url, timeout=0.5):
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        try:
            models = client.models.list()
            print(f"\nLoaded models on running llama-server ({base_url}):")
            for m in models.data:
                print("  -", m.id)
        except Exception as exc:
            print(f"\nCould not query running llama-server at {base_url}: {exc}")


def resolve_llamacpp_model_path(
    model_name: str,
    custom_path: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Resolve and validate a .gguf model path from explicit path, HF Hub cache, or llama.cpp cache."""
    # 1. Explicit -p / --path wins
    if custom_path:
        resolved = os.path.realpath(os.path.expanduser(custom_path))
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Model checkpoint not found: {resolved}")
        return resolved

    # 2. Direct path passed as -n / --name
    direct = os.path.realpath(os.path.expanduser(model_name))
    if os.path.isfile(direct):
        return direct

    cached = _discover_cached_ggufs(cache_dir)
    if not cached:
        raise FileNotFoundError(
            "No cached .gguf models found in ~/.cache/huggingface/hub or ~/.cache/llama.cpp (and no --path provided)"
        )

    # Normalize HuggingFace spec like "lmstudio-community/gemma-4-E4B-it-GGUF:Q4_K_M"
    repo_part, _, quant_part = model_name.partition(":")
    repo_slug = repo_part.lower()
    repo_underscore = repo_part.replace("/", "_").lower()
    quant_slug = quant_part.lower()

    # Exact filename stem match first (e.g., "gemma-4-E4B-it-Q4_K_M")
    for _, path in cached:
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        if stem == model_name.lower() or stem == repo_underscore:
            return path

    # Match against full "org/repo -> filename.gguf" label + optional quant tag
    matches = [
        path for label, path in cached
        if (repo_slug in label.lower() or repo_underscore in label.lower())
        and (not quant_slug or quant_slug in label.lower())
    ]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(os.path.basename(m) for m in matches)
        raise FileNotFoundError(
            f"Multiple GGUF files matched '{model_name}': {names}. "
            "Specify a quant tag (e.g. ':Q4_K_M') or pass --path."
        )

    raise FileNotFoundError(
        f"No GGUF checkpoint matching '{model_name}' found in cache. "
        "Download it first or pass --path /path/to/model.gguf."
    )


def resolve_workspaces(
    model_name: str,
    extra_workspaces: list[str] | None = None,
    include_cwd: bool = False,
) -> tuple[str, list[str]]:
    """Create the default agent sandbox and return (default_workspace, allowed_workspaces)."""
    safe_model_name = model_name.replace("/", "_")
    default_ws = os.path.realpath(os.path.expanduser(f"~/.agents/{safe_model_name}/my_workspace"))
    os.makedirs(default_ws, exist_ok=True)

    allowed: list[str] = []
    if include_cwd:
        allowed.append(os.path.realpath("."))
    if default_ws not in allowed:
        allowed.append(default_ws)

    if extra_workspaces:
        for w in extra_workspaces:
            resolved = os.path.realpath(os.path.expanduser(w))
            if resolved not in allowed:
                allowed.append(resolved)

    return default_ws, allowed


def validate_workspace(target_dir: str, allowed_workspaces: list[str]) -> str:
    """Ensure target_dir is inside one of allowed_workspaces."""
    resolved = os.path.realpath(os.path.expanduser(target_dir))
    if not any(
        resolved == base or resolved.startswith(base + os.sep)
        for base in allowed_workspaces
    ):
        raise ValueError(
            f"Workspace '{resolved}' is outside allowed workspaces: {allowed_workspaces}"
        )
    return resolved


read_tools_to_policies_dict = {
    BuiltinTools.LIST_DIR: "list_directory",
    BuiltinTools.VIEW_FILE: "view_file",
    BuiltinTools.FIND_FILE: "find_file",
    BuiltinTools.SEARCH_DIR: "search_directory",
}

write_tools_to_policies_dict = {
    BuiltinTools.CREATE_FILE: "create_file",
    BuiltinTools.EDIT_FILE: "edit_file",
    BuiltinTools.RUN_COMMAND: "run_command",
}

LOCAL_SYSTEM_INSTRUCTIONS = (
    "You are a helpful, concise, accurate local coding assistant. "
    "Use `list_directory` to list files in a folder, `find_file` (e.g., Pattern='*.cpp') to find files by name, "
    "`search_directory` (always pass both `SearchDirectory` and a non-empty `Query` string) to grep text inside files, "
    "`view_file` (read at most 150 lines at a time using `StartLine` and `EndLine` to save context) to read files, "
    "and `create_file` or `edit_file` to write files. "
    "After receiving tool results, always provide a clear, complete final response to the user. "
    "Never guess paths outside the workspace, never repeat the same tool call, and never invent file contents."
)

def make_dedup_tool_hook(max_repeats: int = 2):
    call_counts: dict[tuple[str, str], int] = {}

    def _guard(tool_call) -> HookResult:
        args_key = json.dumps(getattr(tool_call, "args", {}), sort_keys=True, default=str)
        sig = (tool_call.name, args_key)
        count = call_counts.get(sig, 0) + 1
        call_counts[sig] = count
        print(f"  [Tool -> {tool_call.name}] {tool_call.args}", flush=True)
        if count > max_repeats:
            print(f"  [Tool Blocked -> Duplicate {tool_call.name}]", flush=True)
            return HookResult(allow=False)
        return HookResult(allow=True)

    return _guard


def log_tool_done(tool_call) -> None:
    name = getattr(tool_call, "name", "tool")
    args = getattr(tool_call, "args", {})
    print(f"  [Tool Done -> {name}] {args}", flush=True)


def log_tool_error(ctx, *args, **kwargs):
    err = getattr(ctx, "error", None) or (args[-1] if args else "unknown error")
    print(f"  [Tool Error] {err}", flush=True)


async def extract_response_text(response) -> str:
    """Extract text cleanly whether ChatResponse.text is a method, coroutine, or str."""
    import inspect
    raw = (
        response.text()
        if callable(getattr(response, "text", None))
        else getattr(response, "text", response)
    )
    return await raw if inspect.isawaitable(raw) else str(raw)


def _apply_common_overrides(
    config,
    *,
    read_only: bool = False,
    workspaces: list[str] | None = None,
    policies: list | None = None,
    enable_tool_logging: bool = True,
):
    """
    Apply tool restrictions, system prompt, workspaces, and policies after .lightweight().
    Also, attach to hooks to stream output.
    """
    enabled = list(read_tools_to_policies_dict.keys())
    if not read_only:
        enabled.extend(list(write_tools_to_policies_dict.keys()))

    config.capabilities = CapabilitiesConfig(enabled_tools=enabled)
    config.compaction_config = CompactionConfig(token_threshold=10000)
    config.retry_config = RetryConfig(
        model_output_retry=ModelOutputRetryConfig(max_retries=2),
    )

    if workspaces is not None:
        config.workspaces = workspaces
        ws_block = "\n".join(f"  - {w}" for w in workspaces)
        config.system_instructions = (
            f"{LOCAL_SYSTEM_INSTRUCTIONS}\n\n"
            f"Allowed workspace paths (copy these exact strings character-for-character):\n{ws_block}"
        )
    else:
        config.system_instructions = LOCAL_SYSTEM_INSTRUCTIONS
        
    if policies is not None:
        config.policies = policies

    if enable_tool_logging:
        config.hooks = [
            hooks.pre_tool_call_decide(make_dedup_tool_hook()),
            hooks.post_tool_call(log_tool_done),
            hooks.on_tool_error(log_tool_error),
        ]

    return config


def build_agent_policies(
    workspaces: list[str],
    *,
    read_only: bool = False,
    ask_user_handler=None,
) -> list:
    """
    Build workspace sandbox and tool policies.
    - In interactive CLI mode, pass `ask_user_handler` for write tools.
    - In headless MCP mode (`ask_user_handler=None`), write tools are allowed only inside `workspaces`.
    """
    policies = [
        policy.workspace_only(workspaces),
    ]
    read_policies = list(read_tools_to_policies_dict.values())
    policies.extend(policy.allow(p) for p in read_policies)
    if not read_only:
        write_policies = list(write_tools_to_policies_dict.values())
        if ask_user_handler is not None:
            policies.extend(policy.ask_user(p, handler=ask_user_handler) for p in write_policies)
        else:
            policies.extend(policy.allow(p) for p in write_policies)
    return policies


async def confirm_action(tool_call) -> bool:
    print(f"\n[Approval Required] {tool_call.name}: {tool_call.args}")
    answer = await asyncio.to_thread(input, "Allow? [Y/n]: ")
    return answer.strip().lower() not in ("n", "no")


ask_user_cli_handler: AskUserHandler = confirm_action


async def send_and_print(agent: Agent, prompt: str) -> None:
    print("\nAgent > ", end="", flush=True)
    response = await agent.chat(prompt)
    in_thought = False
    tag_buf = ""

    def _emit(piece: str, force_thought: bool = False) -> None:
        nonlocal in_thought
        if not piece:
            return
        if force_thought:
            if not in_thought:
                print("\n[Thinking] ", end="", flush=True)
                in_thought = True
            print(piece, end="", flush=True)
        else:
            if in_thought:
                print("\n\nAgent > ", end="", flush=True)
                in_thought = False
            print(piece, end="", flush=True)

    try:
        stream = response.chunks() if callable(response.chunks) else response.chunks
        async for chunk in stream:
            thought = getattr(chunk, "thought", None) or getattr(chunk, "thinking", None)
            text = getattr(chunk, "text", None) or getattr(chunk, "delta", None)
            is_thought_flag = bool(getattr(chunk, "is_thought", False))

            if thought or (text and is_thought_flag):
                _emit(thought or text, force_thought=True)
                continue

            raw = text if text is not None else (chunk if isinstance(chunk, str) else "")
            if not raw:
                continue

            tag_buf += raw
            while tag_buf:
                if not in_thought:
                    idx_a = tag_buf.find("<thought>")
                    idx_b = tag_buf.find("<|channel>thought")
                    candidates = [i for i in (idx_a, idx_b) if i != -1]
                    if not candidates:
                        # Keep trailing partial tag prefix if present
                        keep = 17 if "<" in tag_buf[-17:] else 0
                        emit_part = tag_buf[:-keep] if keep else tag_buf
                        _emit(emit_part, force_thought=False)
                        tag_buf = tag_buf[-keep:] if keep else ""
                        break
                    idx = min(candidates)
                    _emit(tag_buf[:idx], force_thought=False)
                    tag_len = len("<thought>") if idx == idx_a else len("<|channel>thought")
                    tag_buf = tag_buf[idx + tag_len:]
                    in_thought = True
                    print("\n[Thinking] ", end="", flush=True)
                else:
                    idx_a = tag_buf.find("</thought>")
                    idx_b = tag_buf.find("<channel|>")
                    candidates = [i for i in (idx_a, idx_b) if i != -1]
                    if not candidates:
                        keep = 10 if "<" in tag_buf[-10:] else 0
                        emit_part = tag_buf[:-keep] if keep else tag_buf
                        if emit_part:
                            print(emit_part, end="", flush=True)
                        tag_buf = tag_buf[-keep:] if keep else ""
                        break
                    idx = min(candidates)
                    if idx > 0:
                        print(tag_buf[:idx], end="", flush=True)
                    tag_len = len("</thought>") if idx == idx_a else len("<channel|>")
                    tag_buf = tag_buf[idx + tag_len:]
                    in_thought = False
                    print("\n\nAgent > ", end="", flush=True)

        if tag_buf:
            _emit(tag_buf, force_thought=in_thought)
        print("\n")
    except KeyboardInterrupt:
        if callable(getattr(response, "cancel", None)):
            await response.cancel()
        print("\n[Generation cancelled by user]\n")


async def run_agent_session(
    config,
    prompt: str | None,
    chat_mode: bool = False,
) -> None:
    async with Agent(config) as agent:
        if prompt:
            await send_and_print(agent, prompt)

        if chat_mode:
            print("Interactive chat started (type 'exit', 'quit', or Ctrl+D to stop).")
            while True:
                try:
                    user_input = await asyncio.to_thread(input, "You > ")
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting chat...")
                    break

                cleaned = user_input.strip()
                if not cleaned:
                    continue
                if cleaned.lower() in ("exit", "quit", "/exit", "/quit"):
                    print("Exiting chat...")
                    break

                await send_and_print(agent, cleaned)


def build_local_openai_agent_config(
    model_name: str,
    *,
    base_url: str,
    api_key: str = "local",
    read_only: bool = False,
    workspaces: list[str] | None = None,
    policies: list | None = None,
    enable_tool_logging: bool = True,
) -> LocalOpenAIAgentConfig:
    """Construct a LocalOpenAIAgentConfig pointing at a local OpenAI-compatible server."""
    config = LocalOpenAIAgentConfig(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    ).lightweight()
    return _apply_common_overrides(
        config,
        read_only=read_only,
        workspaces=workspaces,
        policies=policies,
        enable_tool_logging=enable_tool_logging,
    )


def _is_openai_server_ready(
    base_url: str,
    timeout: float = 1.0,
    expected_model: str | None = None,
    health_paths: tuple[str, ...] = ("/health", "/v1/models"),
) -> bool:
    """Return True if an OpenAI-compatible server responds with HTTP 200 (and matches expected_model if provided)."""
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    if expected_model:
        try:
            with urllib.request.urlopen(f"{origin}/v1/models", timeout=timeout) as resp:
                if resp.status != 200:
                    return False
                payload = json.loads(resp.read().decode("utf-8"))
                loaded_ids = {m.get("id") for m in payload.get("data", [])}
                return expected_model in loaded_ids
        except Exception:
            return False

    for path in health_paths:
        try:
            with urllib.request.urlopen(f"{origin}{path}", timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


def _start_openai_subprocess_server(
    *,
    cmd: list[str],
    base_url: str,
    log_path: str,
    server_label: str,
    expected_model: str | None = None,
    device: str = "vulkan",
    health_paths: tuple[str, ...] = ("/health", "/v1/models"),
    startup_timeout: float = 120.0,
) -> subprocess.Popen:
    """Spawn a local OpenAI-compatible server process and wait until it is ready."""
    expanded_log = os.path.realpath(os.path.expanduser(log_path))
    os.makedirs(os.path.dirname(expanded_log), exist_ok=True)
    log_file = open(expanded_log, "a", encoding="utf-8")

    env = os.environ.copy()
    if device == "cpu":
        env["VK_DRIVER_FILES"] = ""

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_file.close()

    atexit.register(_terminate_proc, proc)

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{server_label} exited early with code {proc.returncode}. Check {expanded_log}"
            )
        if _is_openai_server_ready(
            base_url,
            timeout=0.5,
            expected_model=expected_model,
            health_paths=health_paths,
        ):
            return proc
        time.sleep(0.5)

    _terminate_proc(proc)
    raise TimeoutError(
        f"{server_label} did not become ready at {base_url} within {startup_timeout}s. Check {expanded_log}"
    )


def _terminate_proc(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _is_litertlm_server_ready(
    base_url: str = litertlm_server_url_default,
    timeout: float = 1.0,
    expected_model: str | None = None,
) -> bool:
    """Return True if `litert-lm serve` is ready at `base_url`."""
    return _is_openai_server_ready(
        base_url,
        timeout=timeout,
        expected_model=expected_model,
        health_paths=("/v1/models",),
    )


def _is_llama_server_ready(
    base_url: str = llama_server_url_default,
    timeout: float = 1.0,
    expected_model: str | None = None,
) -> bool:
    """Return True if llama-server is ready at `base_url`."""
    return _is_openai_server_ready(
        base_url,
        timeout=timeout,
        expected_model=expected_model,
        health_paths=("/health", "/v1/models"),
    )


def start_litertlm_server(
    *,
    base_url: str = litertlm_server_url_default,
    expected_model: str | None = None,
    device: str | None = None,
    server_bin: str = "litert-lm",
    config_path: str | None = None,
    startup_timeout: float = 120.0,
) -> subprocess.Popen | None:
    """
    Ensure `litert-lm serve` is running at `base_url` with `expected_model` available.
    """
    if _is_litertlm_server_ready(base_url, expected_model=expected_model):
        return None

    resolved_bin = shutil.which(os.path.expanduser(server_bin))
    if not resolved_bin:
        raise FileNotFoundError(
            f"'{server_bin}' not found in PATH. Pass server_bin='/path/to/litert-lm' or add it to PATH."
        )

    parsed = urlparse(base_url)
    host = parsed.hostname or litertlm_server_host_default
    port = parsed.port or int(litertlm_server_port_default)

    cmd: list[str] = [
        resolved_bin,
        "serve",
        "--host", host,
        "--port", str(port),
    ]
    if config_path:
        cmd.extend(["--config", os.path.realpath(os.path.expanduser(config_path))])

    return _start_openai_subprocess_server(
        cmd=cmd,
        base_url=base_url,
        log_path="~/.litert-lm/litert-lm-server.log",
        server_label="litert-lm serve",
        expected_model=expected_model,
        device=device,
        health_paths=("/v1/models",),
        startup_timeout=startup_timeout,
    )


@contextmanager
def managed_litertlm_server(**kwargs):
    """Context manager wrapper around `start_litertlm_server` for deterministic shutdown."""
    proc = start_litertlm_server(**kwargs)
    try:
        yield proc
    finally:
        _terminate_proc(proc)


def start_llama_server(
    *,
    base_url: str = llama_server_url_default,
    model_path: str | None = None,
    model_alias: str | None = None,
    hf_repo: str | None = None,
    models_dir: str | None = None,
    device: str = "vulkan",
    gpu_device: str | None = None,
    api_key: str | None = None,
    server_bin: str = "llama-server",
    n_gpu_layers: int | str = "all",
    ctx_size: int = 16384,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    flash_attn: str = "on",
    load_mode: str = "none",
    offline: bool = True,
    enable_vision: bool = False,
    extra_args: list[str] | None = None,
    startup_timeout: float = 120.0,
) -> subprocess.Popen | None:
    """Ensure llama-server is running at `base_url` with `model_alias` loaded."""
    if _is_llama_server_ready(base_url, expected_model=model_alias):
        return None

    resolved_bin = shutil.which(os.path.expanduser(server_bin))
    if not resolved_bin:
        raise FileNotFoundError(
            f"'{server_bin}' not found in PATH. Pass server_bin='/path/to/llama-server' or add it to PATH."
        )

    parsed = urlparse(base_url)
    host = parsed.hostname or llama_server_host_default
    port = parsed.port or int(llama_server_port_default)
    effective_ngl = str(n_gpu_layers) if device in ("vulkan", "cuda") else "0"

    cmd: list[str] = [
        resolved_bin,
        "--host", host,
        "--port", str(port),
        "-ngl", effective_ngl,
        "-fit", "off",
        "-np", "1",
        "-c", str(ctx_size),
        "-fa", flash_attn,
        "-ctk", cache_type_k,
        "-ctv", cache_type_v,
        "-lm", load_mode,
        "--reasoning-format", "deepseek",
        "--no-ui",
    ]
    if device == "cpu":
        cmd.extend(["-dev", "none"])
    elif gpu_device:
        cmd.extend(["-dev", gpu_device])
    if offline:
        cmd.append("--offline")
    if not enable_vision:
        cmd.append("--no-mmproj")
    if model_path:
        cmd.extend(["-m", os.path.realpath(os.path.expanduser(model_path))])
    elif hf_repo:
        cmd.extend(["-hf", hf_repo])
    elif models_dir:
        cmd.extend(["--models-dir", os.path.realpath(os.path.expanduser(models_dir))])
    if model_alias:
        cmd.extend(["--alias", model_alias])
    if api_key and api_key != "local":
        cmd.extend(["--api-key", api_key])
    if extra_args:
        cmd.extend(extra_args)

    return _start_openai_subprocess_server(
        cmd=cmd,
        base_url=base_url,
        log_path="~/.cache/llama.cpp/llama-server.log",
        server_label="llama-server",
        expected_model=model_alias,
        device=device,
        health_paths=("/health", "/v1/models"),
        startup_timeout=startup_timeout,
    )


@contextmanager
def managed_llama_server(**kwargs):
    """Context manager wrapper around `start_llama_server` for deterministic shutdown."""
    proc = start_llama_server(**kwargs)
    try:
        yield proc
    finally:
        _terminate_proc(proc)

