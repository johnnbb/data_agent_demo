import json
import pytest
from langgraph.graph import END

from app.graph.spec import NodeName, StateKey
from app.graph.edges import feasibility_assessment_edge
from app.graph.plan_models import Plan, ExecutionStep, ToolParameters
from app.graph.nodes.feasibility_assessment import (
    FeasibilityAssessmentNode,
    feasibility_assessment_node,
)
from app.graph.nodes.planner import PlannerNode, planner_node


def test_feasibility_assessment_edge_logic():
    """测试可行性评估条件路由边的分流逻辑"""
    # 1. 判定为《数据分析》时，放行进入 PLANNER_NODE
    state_feasible = {
        StateKey.FEASIBILITY_RESULT.value: "【需求类型】：《数据分析》\n【语种类型】：《中文》\n【需求内容】：查询虚拟学校的 SAT 数学成绩"
    }
    assert feasibility_assessment_edge(state_feasible) == NodeName.PLANNER.value

    # 2. 判定为《需要澄清》时，路由至 END
    state_clarify = {
        StateKey.FEASIBILITY_RESULT.value: "【需求类型】：《需要澄清》\n【语种类型】：《中文》\n【需求内容】：请问高价值客户的定义是什么？"
    }
    assert feasibility_assessment_edge(state_clarify) == END

    # 3. 判定为《自由闲聊》时，路由至 END
    state_chitchat = {
        StateKey.FEASIBILITY_RESULT.value: "【需求类型】：《自由闲聊》\n【语种类型】：《中文》\n【需求内容】：你好，今天天气不错"
    }
    assert feasibility_assessment_edge(state_chitchat) == END


def test_feasibility_assessment_node_execution():
    """测试 FeasibilityAssessmentNode 端到端真实大模型研判"""
    state = {
        "input": "What is the highest percentage of K-12 students eligible for free meals among schools in Alameda County?",
        "databaseId": "california_schools",
        StateKey.REWRITE_QUERY.value: "What is the highest percentage of K-12 students eligible for free meals among schools in Alameda County?",
        StateKey.EVIDENCE.value: "Alameda County refers to County = 'Alameda'; K-12 free meals refers to Percent (%) Eligible Free (K-12)",
        StateKey.TABLE_RELATION.value: {
            "databaseId": "california_schools",
            "dbTables": [
                {
                    "name": "frpm",
                    "columns": [
                        {"name": "CDSCode", "type": "character varying", "isPrimaryKey": True},
                        {"name": "Percent (%) Eligible Free (K-12)", "type": "double precision"},
                        {"name": "County", "type": "character varying"},
                    ],
                }
            ],
            "dbForeignKeys": [],
        },
    }

    result = feasibility_assessment_node(state)
    assert StateKey.FEASIBILITY_RESULT.value in result
    output = result[StateKey.FEASIBILITY_RESULT.value]
    assert "【需求类型】" in output
    assert "《数据分析》" in output


def test_planner_node_execution():
    """测试 PlannerNode 生成结构化可执行计划"""
    state = {
        "input": "How many schools that are exclusively virtual have an average SAT Math score greater than 400?",
        "databaseId": "california_schools",
        StateKey.REWRITE_QUERY.value: "How many schools that are exclusively virtual have an average SAT Math score greater than 400?",
        StateKey.EVIDENCE.value: "exclusively virtual refers to Virtual = 'F' or Virtual = 'C'",
        StateKey.TABLE_RELATION.value: {
            "databaseId": "california_schools",
            "dbTables": [
                {
                    "name": "schools",
                    "columns": [
                        {"name": "CDSCode", "type": "character varying", "isPrimaryKey": True},
                        {"name": "School", "type": "character varying"},
                        {"name": "Virtual", "type": "character varying"},
                    ],
                },
                {
                    "name": "satscores",
                    "columns": [
                        {"name": "cds", "type": "character varying"},
                        {"name": "AvgScrMath", "type": "integer"},
                    ],
                },
            ],
            "dbForeignKeys": [
                {
                    "sourceColumn": {"name": "cds", "dbTable": {"name": "satscores"}},
                    "targetColumn": {"name": "CDSCode", "dbTable": {"name": "schools"}},
                }
            ],
        },
    }

    result = planner_node(state)
    assert StateKey.PLAN.value in result

    plan_str = result[StateKey.PLAN.value]
    # 验证能成功反序列化为 Plan 强类型模型
    plan_obj = Plan.model_validate_json(plan_str)
    assert len(plan_obj.thought_process) > 0
    assert len(plan_obj.execution_plan) >= 1

    # 验证第一步应为 SQL_GENERATE_NODE
    first_step = plan_obj.execution_plan[0]
    assert first_step.step == 1
    assert first_step.tool_to_use in [
        "SQL_GENERATE_NODE",
        "PYTHON_GENERATE_NODE",
    ]
    assert first_step.tool_parameters.instruction is not None

