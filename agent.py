"""Single-file Harbor agent harness: --agent-import-path agent:AutoAgent."""

from __future__ import annotations

import json
import shlex
import time
from datetime import datetime, timezone

from agents import Agent, Runner, function_tool
from agents.items import (
    ItemHelpers,
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.tool import FunctionTool
from agents.usage import Usage
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


# ============================================================================
# EDITABLE HARNESS — prompt, tools, agent construction
# ============================================================================

SYSTEM_PROMPT = """You are an autonomous coding and terminal agent.

When working on model-ingest tasks, optimize for staged correctness before
performance. Read the task instructions, establish the first failing gate, make
one focused intervention, run the cheapest validation that can confirm it, and
record useful findings in the task's work directory when one is provided. Never
claim success from intent; verify the produced artifacts and scores.
"""
MODEL = "gpt-5"
MAX_TURNS = 80


def create_tools(environment: BaseEnvironment) -> list[FunctionTool]:
    """Create tools for the agent. Add new tools here."""

    async def _exec(command: str, timeout_sec: int = 120) -> str:
        try:
            timeout = max(1, min(int(timeout_sec), 3600))
            result = await environment.exec(command=command, timeout_sec=timeout)
            out = ""
            if result.stdout:
                out += result.stdout
            if result.stderr:
                out += f"\nSTDERR:\n{result.stderr}" if out else f"STDERR:\n{result.stderr}"
            return out or "(no output)"
        except Exception as exc:
            return f"ERROR: {exc}"

    @function_tool
    async def run_shell(command: str, timeout_sec: int = 120) -> str:
        """Run a shell command in the task environment. Returns stdout and stderr."""
        return await _exec(command, timeout_sec)

    @function_tool
    async def list_files(path: str = ".", max_depth: int = 3, limit: int = 200) -> str:
        """List files below a path with bounded depth and output length."""
        depth = max(1, min(int(max_depth), 8))
        max_items = max(1, min(int(limit), 1000))
        command = (
            f"find {shlex.quote(path)} -maxdepth {depth} -type f "
            f"| sort | head -n {max_items}"
        )
        return await _exec(command, timeout_sec=60)

    @function_tool
    async def read_text_file(path: str, start_line: int = 1, max_lines: int = 200) -> str:
        """Read a bounded line range from a text file in the task environment."""
        start = max(1, int(start_line))
        lines = max(1, min(int(max_lines), 1000))
        command = (
            f"python3 - {shlex.quote(path)} {start} {lines} <<'PY'\n"
            "from pathlib import Path\n"
            "import sys\n"
            "path = Path(sys.argv[1])\n"
            "start = int(sys.argv[2])\n"
            "limit = int(sys.argv[3])\n"
            "try:\n"
            "    with path.open('r', encoding='utf-8', errors='replace') as f:\n"
            "        for i, line in enumerate(f, start=1):\n"
            "            if i < start:\n"
            "                continue\n"
            "            if i >= start + limit:\n"
            "                break\n"
            "            print(f'{i}: {line}', end='')\n"
            "except Exception as exc:\n"
            "    print(f'ERROR: {exc}')\n"
            "PY"
        )
        return await _exec(command, timeout_sec=60)

    @function_tool
    async def summarize_json(path: str) -> str:
        """Summarize a JSON artifact without dumping the whole file."""
        command = (
            f"python3 - {shlex.quote(path)} <<'PY'\n"
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "\n"
            "def summarize(value, depth=0):\n"
            "    if depth >= 3:\n"
            "        return type(value).__name__\n"
            "    if isinstance(value, dict):\n"
            "        out = {}\n"
            "        for key, item in list(value.items())[:20]:\n"
            "            if isinstance(item, (dict, list)):\n"
            "                out[key] = summarize(item, depth + 1)\n"
            "            else:\n"
            "                out[key] = item\n"
            "        if len(value) > 20:\n"
            "            out['...'] = f'{len(value) - 20} more keys'\n"
            "        return out\n"
            "    if isinstance(value, list):\n"
            "        return {'length': len(value), 'sample': [summarize(v, depth + 1) for v in value[:5]]}\n"
            "    return value\n"
            "\n"
            "try:\n"
            "    data = json.loads(Path(sys.argv[1]).read_text())\n"
            "    print(json.dumps(summarize(data), indent=2))\n"
            "except Exception as exc:\n"
            "    print(f'ERROR: {exc}')\n"
            "PY"
        )
        return await _exec(command, timeout_sec=60)

    @function_tool
    async def score_tron_ingest(
        gates_json: str = "",
        architecture_json: str = "",
        eqsat_structure_json: str = "",
        typedfx_logits_json: str = "",
        bulk_logits_json: str = "",
        tokens_json: str = "",
        performance_json: str = "",
        output_json: str = "/logs/verifier/reward.json",
        output_txt: str = "/logs/reward.txt",
    ) -> str:
        """Compute alpha/tau/delta reward from Tron ingest artifacts."""
        args = []
        if gates_json:
            args += ["--gates", gates_json]
        if architecture_json:
            args += ["--architecture", architecture_json]
        if eqsat_structure_json:
            args += ["--eqsat-structure", eqsat_structure_json]
        if typedfx_logits_json:
            args += ["--typedfx-logits", typedfx_logits_json]
        if bulk_logits_json:
            args += ["--bulk-logits", bulk_logits_json]
        if tokens_json:
            args += ["--tokens", tokens_json]
        if performance_json:
            args += ["--performance", performance_json]
        args += ["--output-json", output_json, "--output-txt", output_txt, "--print-json"]
        command = "python3 -m tron_ingest_autoagent.score " + " ".join(shlex.quote(arg) for arg in args)
        return await _exec(command, timeout_sec=120)

    return [run_shell, list_files, read_text_file, summarize_json, score_tron_ingest]


def create_agent(environment: BaseEnvironment) -> Agent:
    """Build the agent. Modify to add handoffs, sub-agents, or agent-as-tool."""
    tools = create_tools(environment)
    return Agent(
        name="autoagent",
        instructions=SYSTEM_PROMPT,
        tools=tools,
        model=MODEL,
    )


async def run_task(
    environment: BaseEnvironment,
    instruction: str,
) -> tuple[object, int]:
    """Run the agent on a task and return (result, duration_ms)."""
    agent = create_agent(environment)
    t0 = time.time()
    result = await Runner.run(agent, input=instruction, max_turns=MAX_TURNS)
    duration_ms = int((time.time() - t0) * 1000)
    return result, duration_ms


# ============================================================================
# FIXED ADAPTER BOUNDARY: do not modify unless the human explicitly asks.
# Harbor integration and trajectory serialization live here.
# ============================================================================

def to_atif(result: object, model: str, duration_ms: int = 0) -> dict:
    """Convert OpenAI Agents SDK RunResult to an ATIF trajectory dict."""
    steps: list[dict] = []
    step_id = 0
    now = datetime.now(timezone.utc).isoformat()

    def _step(source: str, message: str, **extra: object) -> dict:
        nonlocal step_id
        step_id += 1
        step = {
            "step_id": step_id,
            "timestamp": now,
            "source": source,
            "message": message,
        }
        step.update({key: value for key, value in extra.items() if value is not None})
        return step

    pending_tool_call = None
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text:
                steps.append(_step("agent", text, model_name=model))
        elif isinstance(item, ReasoningItem):
            summaries = getattr(item.raw_item, "summary", None)
            reasoning = "\n".join(s.text for s in summaries if hasattr(s, "text")) if summaries else None
            if reasoning:
                steps.append(
                    _step(
                        "agent",
                        "(thinking)",
                        reasoning_content=reasoning,
                        model_name=model,
                    )
                )
        elif isinstance(item, ToolCallItem):
            raw = item.raw_item
            if hasattr(raw, "name"):
                pending_tool_call = raw
        elif isinstance(item, ToolCallOutputItem) and pending_tool_call:
            arguments = (
                json.loads(pending_tool_call.arguments)
                if isinstance(pending_tool_call.arguments, str)
                else pending_tool_call.arguments
            )
            output_str = str(item.output) if item.output else ""
            steps.append(
                _step(
                    "agent",
                    f"Tool: {pending_tool_call.name}",
                    tool_calls=[
                        {
                            "tool_call_id": pending_tool_call.call_id,
                            "function_name": pending_tool_call.name,
                            "arguments": arguments,
                        }
                    ],
                    observation={
                        "results": [
                            {
                                "source_call_id": pending_tool_call.call_id,
                                "content": output_str,
                            }
                        ]
                    },
                )
            )
            pending_tool_call = None

    if pending_tool_call:
        arguments = (
            json.loads(pending_tool_call.arguments)
            if isinstance(pending_tool_call.arguments, str)
            else pending_tool_call.arguments
        )
        steps.append(
            _step(
                "agent",
                f"Tool: {pending_tool_call.name}",
                tool_calls=[
                    {
                        "tool_call_id": pending_tool_call.call_id,
                        "function_name": pending_tool_call.name,
                        "arguments": arguments,
                    }
                ],
            )
        )

    if not steps:
        steps.append(_step("user", "(empty)"))

    usage = Usage()
    for response in result.raw_responses:
        usage.add(response.usage)

    return {
        "schema_version": "ATIF-v1.6",
        "session_id": getattr(result, "last_response_id", None) or "unknown",
        "agent": {"name": "autoagent", "version": "0.1.0", "model_name": model},
        "steps": steps,
        "final_metrics": {
            "total_prompt_tokens": usage.input_tokens,
            "total_completion_tokens": usage.output_tokens,
            "total_cached_tokens": getattr(usage.input_tokens_details, "cached_tokens", 0) or 0,
            "total_cost_usd": None,
            "total_steps": len(steps),
            "extra": {"duration_ms": duration_ms, "num_turns": len(result.raw_responses)},
        },
    }


class AutoAgent(BaseAgent):
    """Harbor agent adapter. Runs the OpenAI agent host-side and proxies shell into the container."""

    SUPPORTS_ATIF = True

    def __init__(self, *args, extra_env: dict[str, str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_env = dict(extra_env) if extra_env else {}

    @staticmethod
    def name() -> str:
        return "autoagent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await environment.exec(command="mkdir -p /task")
        instr_file = self.logs_dir / "instruction.md"
        instr_file.write_text(instruction)
        await environment.upload_file(source_path=instr_file, target_path="/task/instruction.md")

        result, duration_ms = await run_task(environment, instruction)

        atif = to_atif(result, model=MODEL, duration_ms=duration_ms)
        traj_path = self.logs_dir / "trajectory.json"
        traj_path.write_text(json.dumps(atif, indent=2))

        try:
            final_metrics = atif.get("final_metrics", {})
            context.n_input_tokens = final_metrics.get("total_prompt_tokens", 0)
            context.n_output_tokens = final_metrics.get("total_completion_tokens", 0)
            context.n_cache_tokens = final_metrics.get("total_cached_tokens", 0)
        except Exception:
            pass

        usage = Usage()
        for response in result.raw_responses:
            usage.add(response.usage)
        print(
            f"turns={len(result.raw_responses)} duration_ms={duration_ms} "
            f"input={usage.input_tokens} output={usage.output_tokens}"
        )


__all__ = ["AutoAgent"]
