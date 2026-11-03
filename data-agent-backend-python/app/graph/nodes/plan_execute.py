import logging
from typing import Any
from langgraph.graph import END

from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.graph.plan_models import Plan

logger = logging.getLogger(__name__)


class PlanExecuteNode:
    """
    执行调度控制节点。
    依据规划方案中的多步执行计划（execution_plan），按步推进当前步骤号（CURRENT_STEP），
    并将 NEXT_NODE 导向当前步骤指定的执行工具（如 SQL_GENERATE_NODE）。
    若全部步骤执行完毕，重置步骤号并走向终止 END。
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        plan_raw = state.get(StateKey.PLAN.value)
        if not plan_raw:
            logger.warning("[PlanExecuteNode] No plan found in state, routing to END.")
            return {
                StateKey.CURRENT_STEP.value: 1,
                StateKey.NEXT_NODE.value: END,
            }

        try:
            if isinstance(plan_raw, Plan):
                plan_obj = plan_raw
            elif isinstance(plan_raw, dict):
                plan_obj = Plan.model_validate(plan_raw)
            else:
                plan_obj = Plan.model_validate_json(str(plan_raw))
        except Exception as e:
            logger.error(f"[PlanExecuteNode] Failed to parse plan from state: {e}")
            return {
                StateKey.CURRENT_STEP.value: 1,
                StateKey.NEXT_NODE.value: END,
            }

        current_step = state.get(StateKey.CURRENT_STEP.value, 1) or 1
        steps = plan_obj.execution_plan

        if current_step > len(steps):
            logger.info(
                f"[PlanExecuteNode] Plan complete. current_step={current_step}, total_steps={len(steps)}"
            )
            return {
                StateKey.CURRENT_STEP.value: 1,
                StateKey.NEXT_NODE.value: END,
            }

        target_tool = steps[current_step - 1].tool_to_use
        logger.info(
            f"[PlanExecuteNode] Executing step {current_step}/{len(steps)}: dispatching to tool '{target_tool}'"
        )
        return {
            StateKey.NEXT_NODE.value: target_tool,
        }


plan_execute_node = PlanExecuteNode()

