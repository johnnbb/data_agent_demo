import logging
from typing import Any
from langgraph.graph import END

from app.graph.state import AgentState
from app.graph.spec import StateKey, NodeName

logger = logging.getLogger(__name__)


class HumanFeedbackNode:
    """
    人工反馈接入节点（对应 Kotlin 的 HumanFeedbackNode.kt）。
    接收用户在前端提交的确认状态（approved: bool）与修改反馈意见（feedback: str）。
    - 修复次数 >= 3 时，强制终止走向 END。
    - 用户确认（approved=True）时，流转至下一步执行控制节点 PLAN_EXECUTE_NODE。
    - 用户驳回（approved=False）时，将反馈内容注入状态，并回流给 PLANNER_NODE 进行自愈修正，修复计数加 1。
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        count = state.get(StateKey.PLAN_REPAIR_COUNT.value, 1)
        if count >= 3:
            logger.warning(
                f"[HumanFeedbackNode] Plan repair count ({count}) reached limit (3), routing to END."
            )
            return {
                StateKey.HUMAN_NEXT_NODE.value: END,
            }

        approved = state.get(StateKey.CONFIRMATION_APPROVED.value, False)
        if approved:
            logger.info(
                "[HumanFeedbackNode] User approved the execution plan. Routing to PLAN_EXECUTE_NODE."
            )
            return {
                StateKey.HUMAN_NEXT_NODE.value: NodeName.PLAN_EXECUTION.value,
            }
        else:
            feedback_content = state.get(StateKey.CONFIRMATION_FEEDBACK.value, "") or ""
            feedback_str = (
                feedback_content.strip()
                if isinstance(feedback_content, str)
                else str(feedback_content)
            )
            final_feedback = feedback_str if feedback_str else "Plan rejected by user"
            logger.info(
                f"[HumanFeedbackNode] User rejected the plan. Routing back to PLANNER_NODE. "
                f"Repair count: {count + 1}, Feedback: '{final_feedback}'"
            )
            return {
                StateKey.HUMAN_NEXT_NODE.value: NodeName.PLANNER.value,
                StateKey.PLAN_REPAIR_COUNT.value: count + 1,
                StateKey.CONFIRMATION_FEEDBACK.value: final_feedback,
                StateKey.PLAN_VALIDATION_ERROR.value: final_feedback,
            }


human_feedback_node = HumanFeedbackNode()

