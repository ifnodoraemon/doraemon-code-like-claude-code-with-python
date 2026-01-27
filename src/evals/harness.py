"""
Evaluation Harness for Polymath Agent

Provides automated testing and evaluation framework following Anthropic standards:
- Multiple trials per task for statistical significance
- Isolated sandbox environments
- Structured assertions and model-based grading
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from statistics import mean

from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# Use new Google GenAI SDK (consistent with main CLI)
from google import genai
from google.genai import types

from src.core.config import load_config
from src.core.logger import TraceLogger
from src.evals.model_grader import ModelGrader
from src.host.client import MultiServerMCPClient

console = Console()


class EvaluationHarness:
    def __init__(self, dataset_path: str, output_dir: str = "eval_results", n_trials: int = 3):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.n_trials = n_trials  # Anthropic Gap 1: Multiple Trials
        self.grader = ModelGrader()
        self.results = []
        os.makedirs(output_dir, exist_ok=True)

    def load_tasks(self) -> list[dict]:
        try:
            with open(self.dataset_path) as f:
                if self.dataset_path.endswith(".jsonl"):
                    return [json.loads(line) for line in f]
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load dataset from {self.dataset_path}: {e}") from e

    def check_assertions(
        self, assertions: list[dict], trace: list[dict], final_output: str, sandbox_dir: str
    ) -> dict:
        """运行硬指标检查 (支持沙箱路径修正)"""
        passed = True
        reasons = []

        for asm in assertions:
            atype = asm.get("type")

            if atype == "file_exists":
                # Check file existence in SANDBOX
                rel_path = asm["path"]
                full_path = os.path.join(sandbox_dir, rel_path)
                if os.path.exists(full_path):
                    reasons.append(f"✅ File {rel_path} exists")
                else:
                    passed = False
                    reasons.append(f"❌ File {rel_path} not found")

            elif atype == "tool_used":
                tool_name = asm["tool"]
                called = any(e["type"] == "tool_call" and e["name"] == tool_name for e in trace)
                if called:
                    reasons.append(f"✅ Tool {tool_name} called")
                else:
                    passed = False
                    reasons.append(f"❌ Tool {tool_name} NOT called")

            elif atype == "output_contains":
                pattern = asm["pattern"]
                if pattern.lower() in final_output.lower():
                    reasons.append(f"✅ Output matches '{pattern}'")
                else:
                    passed = False
                    reasons.append(f"❌ Output missing '{pattern}'")

        return {"pass": passed, "reasons": reasons}

    async def run_single_trial(self, task: dict, trial_id: int) -> dict:
        """运行单次试验 (Isolated Environment)"""
        tracer = TraceLogger()
        config = load_config()

        # Anthropic Gap 3: Isolation
        # 创建临时沙箱目录，并强制改变当前工作目录
        sandbox_dir = tempfile.mkdtemp(prefix=f"eval_{task.get('id')}_t{trial_id}_")
        original_cwd = os.getcwd()

        # 复制必要的 materials 到沙箱 (如果需要)
        # shutil.copytree("materials", os.path.join(sandbox_dir, "materials"))

        start_time = time.time()
        final_output = ""
        error = None

        try:
            # 切换到沙箱
            try:
                os.chdir(sandbox_dir)
            except OSError as e:
                raise RuntimeError(f"Failed to change to sandbox directory: {e}") from e

            mcp_client = MultiServerMCPClient(tracer=tracer)
            # 注意：这里需要确保 connect_to_config 能在沙箱中找到 server 脚本
            # 由于我们在 client.py 做了绝对路径解析，这里通常没问题
            await mcp_client.connect_to_config(config)

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            client = genai.Client(api_key=api_key)

            # Setup Tools using new SDK
            active_tools = []
            for _server_name, session in mcp_client.sessions.items():
                result = await session.list_tools()
                for tool in result.tools:
                    func_decl = types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description or "",
                        parameters=tool.inputSchema,
                    )
                    active_tools.append(func_decl)

            persona = config.get("persona", {})
            sys_prompt = f"You are {persona.get('name', 'Polymath')}. Answer user request."

            # Create chat with new SDK
            tool_obj = types.Tool(function_declarations=active_tools)
            gen_config = types.GenerateContentConfig(
                tools=[tool_obj],
                system_instruction=sys_prompt,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            model_name = os.getenv("EVAL_MODEL", "gemini-1.5-pro")
            chat = client.chats.create(model=model_name, config=gen_config, history=[])

            # Run Loop
            tracer.log("user_input", "prompt", task["prompt"])
            response = chat.send_message(task["prompt"])

            for _ in range(10):  # Max steps
                if not response.candidates:
                    tracer.log("system_error", "empty_response", "No candidates in response")
                    break

                parts = response.candidates[0].content.parts
                has_tool_call = False

                for part in parts:
                    if part.function_call:
                        has_tool_call = True
                        fc = part.function_call
                        try:
                            # Convert args to dict safely
                            if hasattr(fc.args, "items"):
                                args_dict = dict(fc.args.items())
                            else:
                                try:
                                    args_dict = dict(fc.args)
                                except (TypeError, ValueError):
                                    args_dict = {}

                            res = await mcp_client.call_tool(fc.name, args_dict)
                            # Send function response using new SDK
                            response_part = types.Part.from_function_response(
                                name=fc.name, response={"result": res}
                            )
                            response = chat.send_message([response_part])
                        except Exception as e:
                            tracer.log("system_error", "tool_loop", str(e))
                            break
                    elif part.text:
                        final_output = part.text

                if not has_tool_call:
                    tracer.log("model_output", "final", final_output)
                    break

            await mcp_client.cleanup()

        except Exception as e:
            error = str(e)
            tracer.log("system_error", "fatal", error)
        finally:
            # 恢复环境
            try:
                os.chdir(original_cwd)
            except OSError:
                pass  # Best effort recovery
            # 可选：保留沙箱以供调试，或者删除
            # import shutil
            # try:
            #     shutil.rmtree(sandbox_dir)
            # except Exception as e:
            #     logger.warning(f"Failed to cleanup sandbox {sandbox_dir}: {e}")

        duration = time.time() - start_time

        # Grading
        assertions = task.get("assertions", [])
        # Pass sandbox_dir to checker
        code_grade = self.check_assertions(assertions, tracer.events, final_output, sandbox_dir)

        rubric_grade = {"score": 0, "pass": False}
        if task.get("rubric"):
            rubric_grade = self.grader.grade(task["prompt"], final_output, task["rubric"])

        passed = code_grade["pass"] and (rubric_grade["pass"] if task.get("rubric") else True)

        return {
            "trial_id": trial_id,
            "trace": tracer.export(),
            "output": final_output,
            "code_grade": code_grade,
            "rubric_grade": rubric_grade,
            "passed": passed,
            "error": error,
            "metrics": {
                "duration_s": round(duration, 2),
                "steps": len([e for e in tracer.events if e.type == "tool_call"]),
            },
            "sandbox": sandbox_dir,
        }

    async def run(self):
        tasks = self.load_tasks()
        console.print("[bold]Starting PolyEval v4.0 (Anthropic Standard)[/bold]")
        console.print(f"Tasks: {len(tasks)} | Trials per task: {self.n_trials}")

        run_timestamp = int(time.time())
        summary_table = Table(title="Evaluation Summary")
        summary_table.add_column("Task ID")
        summary_table.add_column("Pass Rate", style="bold")
        summary_table.add_column("Avg Score")
        summary_table.add_column("Avg Steps")
        summary_table.add_column("Avg Duration")

        for task in tasks:
            console.print(f"\n[cyan]--- Evaluating: {task['id']} ---[/cyan]")
            task_results = []

            for i in range(self.n_trials):
                console.print(f"Trial {i + 1}/{self.n_trials}...", end=" ")
                res = await self.run_single_trial(task, i)
                task_results.append(res)

                status = "✅" if res["passed"] else "❌"
                console.print(f"{status} ({res['metrics']['duration_s']}s)")

                # Save trace
                trace_name = f"trace_{run_timestamp}_{task['id']}_trial{i}.json"
                with open(os.path.join(self.output_dir, trace_name), "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)

            # Aggregation
            pass_count = sum(1 for r in task_results if r["passed"])
            pass_rate = (pass_count / self.n_trials) * 100

            avg_score = mean([r["rubric_grade"].get("score", 0) for r in task_results])
            avg_steps = mean([r["metrics"]["steps"] for r in task_results])
            avg_duration = mean([r["metrics"]["duration_s"] for r in task_results])

            color = "green" if pass_rate == 100 else "yellow" if pass_rate > 50 else "red"
            summary_table.add_row(
                task["id"],
                f"[{color}]{pass_rate:.0f}% ({pass_count}/{self.n_trials})[/{color}]",
                f"{avg_score:.1f}",
                f"{avg_steps:.1f}",
                f"{avg_duration:.2f}s",
            )

        console.print("\n")
        console.print(summary_table)


if __name__ == "__main__":
    dataset = "polymath/tests/evals/dataset_v3.json"
    # Default k=2 for fast demo, typically k=10
    harness = EvaluationHarness(dataset, n_trials=2)
    asyncio.run(harness.run())
