import pytest
from langgraph.graph import END

from app.graph.spec import NodeName, StateKey
from app.graph.nodes.human_feedback import HumanFeedbackNode
from app.graph.nodes.plan_execute import PlanExecuteNode
from app.graph.edges import human_feedback_edge
from app.graph.plan_models import Plan, ExecutionStep, ToolParameters
from app.graph.builder import build_graph


def test_human_feedback_node_approved():
    node = HumanFeedbackNode()
    state = {
        StateKey.CONFIRMATION_APPROVED.value: True,
        StateKey.CONFIRMATION_FEEDBACK.value: "Looks good!",
        StateKey.PLAN_REPAIR_COUNT.value: 1,
    }
    result = node(state)
    assert result[StateKey.HUMAN_NEXT_NODE.value] == NodeName.PLAN_EXECUTION.value
    assert StateKey.PLAN_REPAIR_COUNT.value not in result


def test_human_feedback_node_rejected():
    node = HumanFeedbackNode()
    state = {
        StateKey.CONFIRMATION_APPROVED.value: False,
        StateKey.CONFIRMATION_FEEDBACK.value: "Please group by year instead of month",
        StateKey.PLAN_REPAIR_COUNT.value: 1,
    }
    result = node(state)
    assert result[StateKey.HUMAN_NEXT_NODE.value] == NodeName.PLANNER.value
    assert result[StateKey.PLAN_REPAIR_COUNT.value] == 2
    assert result[StateKey.PLAN_VALIDATION_ERROR.value] == "Please group by year instead of month"
    assert result[StateKey.CONFIRMATION_FEEDBACK.value] == "Please group by year instead of month"


def test_human_feedback_node_max_repairs():
    node = HumanFeedbackNode()
    state = {
        StateKey.CONFIRMATION_APPROVED.value: False,
        StateKey.PLAN_REPAIR_COUNT.value: 3,
    }
    result = node(state)
    assert result[StateKey.HUMAN_NEXT_NODE.value] == END


def test_plan_execute_node_step_dispatch():
    node = PlanExecuteNode()
    sample_plan = Plan(
        thought_process="Check california_schools tables",
        execution_plan=[
            ExecutionStep(
                step=1,
                tool_to_use="SQL_GENERATE_NODE",
                tool_parameters=ToolParameters(instruction="Generate SQL query for school data"),
            ),
            ExecutionStep(
                step=2,
                tool_to_use="SQL_EXECUTE_NODE",
                tool_parameters=ToolParameters(instruction="Execute SQL query"),
            ),
        ],
    )
    # Step 1
    state = {
        StateKey.PLAN.value: sample_plan.model_dump_json(),
        StateKey.CURRENT_STEP.value: 1,
    }
    result = node(state)
    assert result[StateKey.NEXT_NODE.value] == "SQL_GENERATE_NODE"

    # Step 2
    state[StateKey.CURRENT_STEP.value] = 2
    result = node(state)
    assert result[StateKey.NEXT_NODE.value] == "SQL_EXECUTE_NODE"

    # Step 3 (all steps finished -> reset and END)
    state[StateKey.CURRENT_STEP.value] = 3
    result = node(state)
    assert result[StateKey.CURRENT_STEP.value] == 1
    assert result[StateKey.NEXT_NODE.value] == END


def test_human_feedback_edge():
    assert human_feedback_edge({StateKey.HUMAN_NEXT_NODE.value: NodeName.PLAN_EXECUTION.value}) == NodeName.PLAN_EXECUTION.value
    assert human_feedback_edge({StateKey.HUMAN_NEXT_NODE.value: NodeName.PLANNER.value}) == NodeName.PLANNER.value
    assert human_feedback_edge({StateKey.HUMAN_NEXT_NODE.value: END}) == END
    assert human_feedback_edge({}) == END


@pytest.mark.asyncio
async def test_graph_interrupt_and_resume_flow():
    """
    测试 LangGraph 中断与人工审核恢复机制：
    1. 初始输入流转至 PLANNER 完成后，在 HUMAN_FEEDBACK 节点前自动中断挂起
    2. 检查 state_snapshot.next 确为 HUMAN_FEEDBACK_NODE
    3. 模拟前端提交 approved=True，以 None 继续恢复执行，最终抵达 END
    """
    graph = build_graph(interrupt_before_feedback=True)
    thread_id = "test-hitl-resume-999"
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "input": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        "databaseId": "california_schools",
    }

    # 1. 首次触发执行，应在 HUMAN_FEEDBACK_NODE 前暂停
    first_pass_nodes = []
    async for mode, event in graph.astream(initial_input, config, stream_mode=["updates"]):
        if mode == "updates":
            first_pass_nodes.extend(list(event.keys()))

    assert NodeName.PLANNER.value in first_pass_nodes
    assert NodeName.HUMAN_FEEDBACK.value not in first_pass_nodes
    assert NodeName.PLAN_EXECUTION.value not in first_pass_nodes

    # 检查状态快照：挂起在 HUMAN_FEEDBACK_NODE
    snapshot = await graph.aget_state(config)
    assert snapshot.next == (NodeName.HUMAN_FEEDBACK.value,)

    # 2. 模拟用户在前端点击“确认继续”
    await graph.aupdate_state(
        config,
        {
            StateKey.CONFIRMATION_APPROVED.value: True,
            StateKey.CONFIRMATION_FEEDBACK.value: "确认通过，立即执行",
        },
    )

    # 3. 恢复图执行
    second_pass_nodes = []
    async for mode, event in graph.astream(None, config, stream_mode=["updates"]):
        if mode == "updates":
            second_pass_nodes.extend(list(event.keys()))

    assert NodeName.HUMAN_FEEDBACK.value in second_pass_nodes
    assert NodeName.PLAN_EXECUTION.value in second_pass_nodes

    # 最终状态已抵达 END，无后续待执行节点
    final_snapshot = await graph.aget_state(config)
    assert final_snapshot.next == ()


@pytest.mark.asyncio
async def test_graph_reject_loop_and_second_interrupt():
    """
    测试用户驳回方案后的自愈重规划链路：
    1. 首次停在 HUMAN_FEEDBACK
    2. 用户提交驳回 (approved=False)
    3. 状态机回流到 PLANNER_NODE，携带反馈重新规划
    4. 重新规划完成后，再次在 HUMAN_FEEDBACK 节点前中断挂起
    5. 用户二次确认通过 (approved=True)，最终流转至 PLAN_EXECUTION 并抵达 END
    """
    graph = build_graph(interrupt_before_feedback=True)
    thread_id = "test-hitl-reject-loop-111"
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "input": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        "databaseId": "california_schools",
    }

    # 1. 首次触发
    async for mode, _ in graph.astream(initial_input, config, stream_mode=["updates"]):
        pass
    snapshot1 = await graph.aget_state(config)
    assert snapshot1.next == (NodeName.HUMAN_FEEDBACK.value,)

    # 2. 模拟用户驳回反馈
    await graph.aupdate_state(
        config,
        {
            StateKey.CONFIRMATION_APPROVED.value: False,
            StateKey.CONFIRMATION_FEEDBACK.value: "请不要按月汇总，改为按季度汇总并包含学校全称",
        },
    )

    # 3. 恢复图：回流至 PLANNER 重新规划
    replan_nodes = []
    async for mode, event in graph.astream(None, config, stream_mode=["updates"]):
        if mode == "updates":
            replan_nodes.extend(list(event.keys()))

    assert NodeName.HUMAN_FEEDBACK.value in replan_nodes
    assert NodeName.PLANNER.value in replan_nodes
    assert NodeName.PLAN_EXECUTION.value not in replan_nodes

    # 4. 再次在 HUMAN_FEEDBACK 节点前挂起
    snapshot2 = await graph.aget_state(config)
    assert snapshot2.next == (NodeName.HUMAN_FEEDBACK.value,)
    assert snapshot2.values.get(StateKey.PLAN_REPAIR_COUNT.value) == 2

    # 5. 用户二次确认通过
    await graph.aupdate_state(
        config,
        {
            StateKey.CONFIRMATION_APPROVED.value: True,
            StateKey.CONFIRMATION_FEEDBACK.value: "方案调整后非常满意，请执行",
        },
    )

    # 6. 二次恢复：执行并完成
    final_nodes = []
    async for mode, event in graph.astream(None, config, stream_mode=["updates"]):
        if mode == "updates":
            final_nodes.extend(list(event.keys()))

    assert NodeName.HUMAN_FEEDBACK.value in final_nodes
    assert NodeName.PLAN_EXECUTION.value in final_nodes

    final_snapshot = await graph.aget_state(config)
    assert final_snapshot.next == ()

