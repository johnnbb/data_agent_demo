import json
import pytest
from unittest.mock import MagicMock
from langgraph.graph import END

from app.graph.spec import NodeName, StateKey
from app.graph.plan_models import Plan, ExecutionStep, ToolParameters
from app.graph.nodes.report_generator import ReportGeneratorNode
from app.graph.nodes.plan_execute import plan_execute_node
from app.graph.edges import plan_executor_edge
from app.graph.builder import build_graph


def test_build_user_requirements_and_plan():
    plan = Plan(
        thought_process="分析加州贫困生分布特征并给出资助建议",
        execution_plan=[
            ExecutionStep(
                step=1,
                tool_to_use=NodeName.SQL_GENERATION.value,
                tool_parameters=ToolParameters(instruction="查询各县最高贫困率学校"),
            ),
            ExecutionStep(
                step=2,
                tool_to_use=NodeName.REPORT_GENERATION.value,
                tool_parameters=ToolParameters(
                    summary_and_recommendations="输出重点关注的高危贫困学校及专项拨款建议"
                ),
            ),
        ],
    )
    user_input = "哪些学校的贫困生比例最高？"
    res = ReportGeneratorNode.build_user_requirements_and_plan(user_input, plan)

    assert "## 用户原始需求" in res
    assert user_input in res
    assert "## 执行计划概述" in res
    assert "分析加州贫困生分布特征" in res
    assert "### 步骤 1: 步骤编号 1" in res
    assert "查询各县最高贫困率学校" in res


def test_build_analysis_steps_and_data_with_sql_and_python():
    plan = Plan(
        thought_process="两步分析",
        execution_plan=[
            ExecutionStep(
                step=1,
                tool_to_use=NodeName.SQL_GENERATION.value,
                tool_parameters=ToolParameters(
                    instruction="查询 Alameda 县前两名学校",
                    sql_query="SELECT CDSCode, School FROM frpm LIMIT 2;",
                ),
            ),
            ExecutionStep(
                step=2,
                tool_to_use=NodeName.PYTHON_GENERATION.value,
                tool_parameters=ToolParameters(instruction="计算统计指标"),
            ),
        ],
    )
    execution_results = {
        "step_1": {
            "column": ["CDSCode", "School"],
            "data": [{"CDSCode": "01", "School": "Lincoln"}],
        },
        "step_2": {"mean": 0.85, "max": 0.95},
        "step_2_analysis": "全县贫困生平均比例达 85%，处于高位警惕区间。",
    }

    res = ReportGeneratorNode.build_analysis_steps_and_data(plan, execution_results)

    # 1. 验证可追溯性：SQL 语句被包含
    assert "SELECT CDSCode, School FROM frpm LIMIT 2;" in res
    assert "```sql" in res

    # 2. 验证数据被包含
    assert "Lincoln" in res
    assert "```json" in res

    # 3. 验证 _analysis 键未作为独立 step 出现，而是贴在 step_2 下方
    assert "### step_2_analysis" not in res
    assert "**Python 分析结果**: 全县贫困生平均比例达 85%" in res


def test_report_generator_node_execution():
    node = ReportGeneratorNode()
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = """# 加州学校贫困率专项分析报告

## 一、核心结论
Alameda 县 Lincoln 学校贫困率处于前列。

```echarts
{
  "title": { "text": "学校贫困率" },
  "series": [{ "type": "bar", "data": [85] }]
}
```
"""
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp
    node.client = mock_client

    plan = Plan(
        thought_process="查询与总结",
        execution_plan=[
            ExecutionStep(
                step=1,
                tool_to_use=NodeName.REPORT_GENERATION.value,
                tool_parameters=ToolParameters(summary_and_recommendations="输出决策建议"),
            )
        ],
    )
    state = {
        StateKey.REWRITE_QUERY.value: "输出加州贫困生报告",
        StateKey.PLAN.value: plan,
        StateKey.CURRENT_STEP.value: 1,
        StateKey.EXECUTION_OUTPUT.value: {
            "step_1": {"data": [{"school": "Lincoln"}]}
        },
    }

    result = node(state)
    assert StateKey.REPORT_RESULT.value in result
    report_md = result[StateKey.REPORT_RESULT.value]
    assert "# 加州学校贫困率专项分析报告" in report_md
    assert "```echarts" in report_md


def test_complete_end_to_end_three_tier_loop_dispatch():
    """
    测试终局完整三层回环调度：
    Step 1: SQL_GENERATION -> (回环) PLAN_EXECUTION
    Step 2: PYTHON_GENERATION -> (回环) PLAN_EXECUTION
    Step 3: REPORT_GENERATION -> END
    """
    plan = Plan(
        thought_process="SQL查数 -> Python计算 -> Report产出",
        execution_plan=[
            ExecutionStep(
                step=1,
                tool_to_use=NodeName.SQL_GENERATION.value,
                tool_parameters=ToolParameters(instruction="SQL查数"),
            ),
            ExecutionStep(
                step=2,
                tool_to_use=NodeName.PYTHON_GENERATION.value,
                tool_parameters=ToolParameters(instruction="Python分析"),
            ),
            ExecutionStep(
                step=3,
                tool_to_use=NodeName.REPORT_GENERATION.value,
                tool_parameters=ToolParameters(summary_and_recommendations="综合报告"),
            ),
        ],
    )

    base_state = {
        StateKey.PLAN.value: plan,
    }

    # Step 1 调度
    s1 = plan_execute_node({**base_state, StateKey.CURRENT_STEP.value: 1})
    assert s1[StateKey.NEXT_NODE.value] == NodeName.SQL_GENERATION.value
    assert plan_executor_edge({**base_state, **s1}) == NodeName.SQL_GENERATION.value

    # Step 2 调度
    s2 = plan_execute_node({**base_state, StateKey.CURRENT_STEP.value: 2})
    assert s2[StateKey.NEXT_NODE.value] == NodeName.PYTHON_GENERATION.value
    assert plan_executor_edge({**base_state, **s2}) == NodeName.PYTHON_GENERATION.value

    # Step 3 调度 (Report 节点)
    s3 = plan_execute_node({**base_state, StateKey.CURRENT_STEP.value: 3})
    assert s3[StateKey.NEXT_NODE.value] == NodeName.REPORT_GENERATION.value
    assert plan_executor_edge({**base_state, **s3}) == NodeName.REPORT_GENERATION.value

    # Step 4 (已全部完成) -> 走向 END
    s4 = plan_execute_node({**base_state, StateKey.CURRENT_STEP.value: 4})
    assert s4[StateKey.NEXT_NODE.value] == END
    assert plan_executor_edge({**base_state, **s4}) == END

