import json
import logging
from typing import Any

from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.shared.python_executor import SimplePythonExecutor, PythonExecutionResult

logger = logging.getLogger(__name__)


class PythonExecuteNode:
    """
    Python 脚本执行节点。
    读取生成的 Python 脚本，以 SQL 查询结果数据作为 stdin 注入，
    在安全沙箱（或隔离子进程）中执行该脚本并捕获结构化输出。
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        python_code = state.get(StateKey.PYTHON_GENERATION_RESULT.value, "") or ""
        logger.info("[PythonExecuteNode] Executing generated Python script...")

        # 提取输入数据
        input_data = []
        sql_exec_result = state.get(StateKey.SQL_EXECUTION_RESULT.value)
        if isinstance(sql_exec_result, dict):
            input_data = sql_exec_result.get("resultSet", {}).get("data", [])
        elif not input_data:
            exec_output = state.get(StateKey.EXECUTION_OUTPUT.value)
            if isinstance(exec_output, dict):
                for key in sorted(exec_output.keys(), reverse=True):
                    val = exec_output[key]
                    if isinstance(val, dict) and "data" in val:
                        input_data = val.get("data", [])
                        break

        input_json_str = json.dumps(input_data, ensure_ascii=False)

        # 执行脚本
        res: PythonExecutionResult = SimplePythonExecutor.execute(
            python_code=python_code,
            input_json=input_json_str,
            timeout_sec=60,
        )

        logger.info(
            f"[PythonExecuteNode] Execution finished. Success: {res.success}, "
            f"Output length: {len(res.output)}, Error length: {len(res.error)}"
        )

        return {
            StateKey.PYTHON_EXECUTION_RESULT.value: res.to_dict(),
        }


python_execute_node = PythonExecuteNode()

