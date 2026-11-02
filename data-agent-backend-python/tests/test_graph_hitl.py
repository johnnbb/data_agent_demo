import pytest
from app.graph.builder import build_graph
from app.graph.spec import NodeName, StateKey


@pytest.mark.asyncio
async def test_dual_recall_graph_execution():
    """
    测试正式的 Text2SQL 双通道召回图：
    START -> EVIDENCE_RECALL_NODE -> SCHEME_RECALL_NODE -> END
    """
    graph = build_graph(interrupt_before_feedback=False)
    thread_id = "test-dual-recall-123"
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "input": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        "databaseId": "california_schools",
    }

    # 执行图流转
    node_events = []
    async for event in graph.astream(initial_input, config, stream_mode="updates"):
        node_events.append(event)

    # 1. 验证经过了五个核心全流程节点
    nodes_executed = [list(e.keys())[0] for e in node_events]
    assert NodeName.EVIDENCE_RECALL.value in nodes_executed
    assert NodeName.SCHEMA_RECALL.value in nodes_executed
    assert NodeName.TABLE_RELATION.value in nodes_executed
    assert NodeName.FEASIBILITY_ASSESSMENT.value in nodes_executed
    assert NodeName.PLANNER.value in nodes_executed

    # 2. 验证最终状态包含了所有前端期望的数据键
    final_state = await graph.aget_state(config)
    values = final_state.values

    assert StateKey.REWRITE_QUERY.value in values
    assert StateKey.EVIDENCE.value in values
    assert StateKey.TABLE_SCHEMA.value in values
    assert StateKey.COLUMN_SCHEMA.value in values
    assert StateKey.TABLE_RELATION.value in values
    assert StateKey.FEASIBILITY_RESULT.value in values
    assert StateKey.PLAN.value in values

    # 3. 验证具体召回、研判与规划数据
    assert "### 业务术语表" in values[StateKey.EVIDENCE.value]
    assert len(values[StateKey.TABLE_SCHEMA.value]) > 0
    assert len(values[StateKey.COLUMN_SCHEMA.value]) > 0
    assert "dbTables" in values[StateKey.TABLE_RELATION.value]
    assert "【需求类型】" in values[StateKey.FEASIBILITY_RESULT.value]
    assert "execution_plan" in values[StateKey.PLAN.value]
