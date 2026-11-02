import json
import pytest
from unittest.mock import MagicMock
from langgraph.graph import END

from app.graph.spec import NodeName, StateKey
from app.graph.plan_models import Plan, ExecutionStep, ToolParameters
from app.graph.nodes.python_generator import PythonGeneratorNode, extract_raw_python
from app.graph.nodes.python_execute import PythonExecuteNode
from app.graph.nodes.python_analyze import PythonAnalyzeNode
from app.shared.python_executor import SimplePythonExecutor, PythonExecutionResult
from app.graph.builder import build_graph


def test_extract_raw_python():
    markdown_code = """```python
import sys
import json
print(json.dumps({"status": "ok"}))
```"""
    cleaned = extract_raw_python(markdown_code)
    assert not cleaned.startswith("```")
    assert not cleaned.endswith("```")
    assert "import sys" in cleaned


def test_simple_python_executor_success():
    script = """
import sys
import json
import pandas as pd

input_data = json.load(sys.stdin)
df = pd.DataFrame(input_data)
summary = {
    "total_rows": len(df),
    "columns": list(df.columns),
    "sum_val": int(df["val"].astype(int).sum())
}
print(json.dumps(summary))
"""
    input_data = json.dumps([{"id": 1, "val": 10}, {"id": 2, "val": 20}])
    res = SimplePythonExecutor.execute(script, input_data, timeout_sec=10)

    assert res.success is True
    assert res.error == ""
    parsed = json.loads(res.output)
    assert parsed["total_rows"] == 2
    assert parsed["sum_val"] == 30


def test_simple_python_executor_error_soft_landing():
    # 模拟语法错误或运行期异常，验证软着陆不崩溃
    script = """
import sys
import json

raise RuntimeError("Custom calculation failure")
"""
    res = SimplePythonExecutor.execute(script, "[]", timeout_sec=5)
    assert res.success is False
    assert "Custom calculation failure" in res.error


def test_simple_python_executor_timeout():
    script = """
import time
time.sleep(3)
"""
    res = SimplePythonExecutor.execute(script, "[]", timeout_sec=1)
    assert res.success is False
    assert "timed out" in res.error.lower()


def test_python_execute_node():
    node = PythonExecuteNode()
    python_code = """
import sys, json, pandas as pd
data = json.load(sys.stdin)
df = pd.DataFrame(data)
res = {"count": len(df), "first_school": df["school_name"].iloc[0]}
print(json.dumps(res, ensure_ascii=False))
"""
    state = {
        StateKey.PYTHON_GENERATION_RESULT.value: python_code,
        StateKey.SQL_EXECUTION_RESULT.value: {
            "resultSet": {
                "column": ["school_name"],
                "data": [{"school_name": "Lincoln High"}, {"school_name": "Washington High"}],
            }
        },
    }

    result = node(state)
    assert StateKey.PYTHON_EXECUTION_RESULT.value in result
    exec_res = result[StateKey.PYTHON_EXECUTION_RESULT.value]
    assert exec_res["success"] is True
    parsed_out = json.loads(exec_res["output"])
    assert parsed_out["count"] == 2
    assert parsed_out["first_school"] == "Lincoln High"


def test_python_analyze_node():
    node = PythonAnalyzeNode()
    # Mock LLM
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "经分析，两所学校表现优秀，其中第一所为 Lincoln High。"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp
    node.client = mock_client

    state = {
        StateKey.REWRITE_QUERY.value: "分析加州表现最好的两所学校",
        StateKey.CURRENT_STEP.value: 2,
        StateKey.PYTHON_EXECUTION_RESULT.value: {
            "success": True,
            "output": json.dumps({"count": 2, "first_school": "Lincoln High"}),
            "error": "",
        },
        StateKey.EXECUTION_OUTPUT.value: {
            "step_1": {"column": ["school_name"], "data": [{"school_name": "Lincoln High"}]}
        },
    }

    result = node(state)
    assert result[StateKey.CURRENT_STEP.value] == 3
    output_map = result[StateKey.EXECUTION_OUTPUT.value]
    assert "step_1" in output_map
    assert "step_2_analysis" in output_map
    assert "Lincoln High" in output_map["step_2_analysis"]


def test_multi_step_loopback_graph_execution():
    """
    验证核心图回环拓扑：
    Step 1: SQL_GENERATE_NODE -> SQL_EXECUTE_NODE -> (回环) PLAN_EXECUTE_NODE
    Step 2: PYTHON_GENERATE_NODE -> PYTHON_EXECUTE_NODE -> PYTHON_ANALYZE_NODE -> END
    """
    graph = build_graph(interrupt_before_feedback=False)

    # 构造两步执行计划
    mock_plan = Plan(
        thought_process="先通过 SQL 查出学校数据，再通过 Python 脚本进行统计分析",
        execution_plan=[
            ExecutionStep(
                step=1,
                tool_to_use=NodeName.SQL_GENERATION.value,
                tool_parameters=ToolParameters(instruction="查询前2所学校的 CDSCode 和 County"),
            ),
            ExecutionStep(
                step=2,
                tool_to_use=NodeName.PYTHON_GENERATION.value,
                tool_parameters=ToolParameters(instruction="统计学校分布并输出总结"),
            ),
        ],
    )

    initial_state = {
        "input": "查询前2所学校并做分布分析",
        "databaseId": "california_schools",
        StateKey.REWRITE_QUERY.value: "查询前2所学校并做分布分析",
        StateKey.FEASIBILITY_RESULT.value: "【需求类型】：《数据分析》",
        StateKey.PLAN.value: mock_plan,
        StateKey.CURRENT_STEP.value: 1,
        # 预设已通过审核，从 PLAN_EXECUTION 开始触发
    }

    # 单独测试调度器回环与工具执行
    from app.graph.nodes.plan_execute import plan_execute_node
    from app.graph.edges import plan_executor_edge

    # 1. Step 1 调度
    step1_dispatch = plan_execute_node(initial_state)
    assert step1_dispatch[StateKey.NEXT_NODE.value] == NodeName.SQL_GENERATION.value
    assert plan_executor_edge({**initial_state, **step1_dispatch}) == NodeName.SQL_GENERATION.value

    # 2. 模拟 SQL 执行完成，更新状态（step 变为 2，写入 step_1 数据）
    from app.graph.nodes.sql_execute import sql_execute_node
    sql_gen_state = {
        **initial_state,
        StateKey.SQL_GENERATION_RESULT.value: 'SELECT "CDSCode", "County" FROM schools LIMIT 2;',
    }
    sql_exec_res = sql_execute_node(sql_gen_state)
    assert sql_exec_res[StateKey.CURRENT_STEP.value] == 2
    assert "step_1" in sql_exec_res[StateKey.EXECUTION_OUTPUT.value]

    # 3. Step 2 回流到调度器 PLAN_EXECUTE_NODE
    state_after_sql = {
        **sql_gen_state,
        **sql_exec_res,
    }
    step2_dispatch = plan_execute_node(state_after_sql)
    assert step2_dispatch[StateKey.NEXT_NODE.value] == NodeName.PYTHON_GENERATION.value
    assert plan_executor_edge({**state_after_sql, **step2_dispatch}) == NodeName.PYTHON_GENERATION.value
