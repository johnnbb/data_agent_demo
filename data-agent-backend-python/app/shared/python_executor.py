import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PythonExecutionResult:
    success: bool
    output: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimplePythonExecutor:
    DEFAULT_DOCKER_IMAGE = "continuumio/anaconda3:latest"
    DEFAULT_MEMORY_LIMIT = "512m"

    @classmethod
    def is_docker_available(cls) -> bool:
        if os.getenv("DATA_AGENT_PYTHON_DISABLE_DOCKER", "false").lower() in ("1", "true", "yes"):
            return False
        try:
            proc = subprocess.run(
                ["docker", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
            )
            return proc.returncode == 0
        except Exception:
            return False

    @classmethod
    def execute(cls, python_code: str, input_json: str, timeout_sec: int = 60) -> PythonExecutionResult:
        work_dir = tempfile.mkdtemp(prefix="ai_python_exec_")
        script_file = os.path.join(work_dir, "script.py")
        data_file = os.path.join(work_dir, "input.json")

        try:
            with open(script_file, "w", encoding="utf-8") as f:
                f.write(python_code)
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(input_json)

            # 1. 尝试 Docker 沙箱执行
            if cls.is_docker_available():
                docker_image = os.getenv("DATA_AGENT_PYTHON_DOCKER_IMAGE", "").strip() or cls.DEFAULT_DOCKER_IMAGE
                memory_limit = os.getenv("DATA_AGENT_PYTHON_MEMORY_LIMIT", "").strip() or cls.DEFAULT_MEMORY_LIMIT

                cmd = [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    "--network",
                    "none",
                    "--cpus",
                    "1",
                    "--memory",
                    memory_limit,
                    "--pids-limit",
                    "128",
                    "--security-opt",
                    "no-new-privileges",
                    "-v",
                    f"{work_dir}:/work:ro",
                    "-w",
                    "/work",
                    docker_image,
                    "python",
                    "/work/script.py",
                ]
                res = cls._run_command(cmd, data_file, timeout_sec)
                if res.success:
                    return res
                else:
                    return PythonExecutionResult(
                        success=False,
                        output=res.output,
                        error=f"{res.error}\n\nSandbox image: {docker_image}",
                    )

            # 2. 优雅降级：本地安全隔离子进程
            logger.info("Docker不可用或未启用，降级为本地隔离子进程执行 Python 脚本")
            cmd = [sys.executable, script_file]
            return cls._run_command(cmd, data_file, timeout_sec, cwd=work_dir)

        except Exception as e:
            logger.exception("执行 Python 脚本发生异常: %s", e)
            return PythonExecutionResult(
                success=False,
                output="",
                error=str(e),
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @classmethod
    def _run_command(
        cls,
        command: list[str],
        input_file: str,
        timeout_sec: int,
        cwd: Optional[str] = None,
    ) -> PythonExecutionResult:
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                input_data = f.read()

            proc = subprocess.run(
                command,
                input=input_data,
                text=True,
                capture_output=True,
                timeout=timeout_sec,
                cwd=cwd,
            )
            return PythonExecutionResult(
                success=(proc.returncode == 0),
                output=proc.stdout.strip(),
                error=proc.stderr.strip(),
            )
        except subprocess.TimeoutExpired:
            return PythonExecutionResult(
                success=False,
                output="",
                error=f"Execution timed out after {timeout_sec} seconds.",
            )
        except Exception as e:
            return PythonExecutionResult(
                success=False,
                output="",
                error=str(e),
            )

