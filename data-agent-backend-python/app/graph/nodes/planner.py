import json
import logging
import re
from typing import Any, Optional
from openai import OpenAI

from app.config import settings
from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.graph.plan_models import Plan
from app.prompts.manager import prompt_manager
from app.retrieval.schema_models import Schema

logger = logging.getLogger(__name__)


class PlannerNode:
    """
    任务规划节点。
    将经可行性确认的业务问题，结合精选后的表结构与业务口径，拆解为结构化多步执行计划。
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
    ):
        self.chat_model = chat_model or settings.CHAT_MODEL
        self.client = OpenAI(
            api_key=openai_api_key or settings.OPENAI_API_KEY,
            base_url=openai_base_url or settings.OPENAI_BASE_URL,
        )

    def __call__(self, state: AgentState) -> dict[str, Any]:
        rewrite_query = state.get(StateKey.REWRITE_QUERY.value) or state.get("input", "")
        evidence = state.get(StateKey.EVIDENCE.value, "")
        schema_data = state.get(StateKey.TABLE_RELATION.value)
        plan_validation_error = state.get(StateKey.PLAN_VALIDATION_ERROR.value) or ""

        if isinstance(schema_data, dict):
            schema_obj = Schema.model_validate(schema_data)
            scheme_prompt = schema_obj.build_scheme_prompt()
        elif isinstance(schema_data, str):
            scheme_prompt = schema_data
        else:
            scheme_prompt = "【无表结构】"

        format_instruction = (
            "你必须输出一个符合以下 JSON Schema 的合法 JSON 对象：\n"
            + json.dumps(Plan.model_json_schema(), ensure_ascii=False)
        )

        prompt = prompt_manager.render(
            "planner.st",
            user_question=rewrite_query,
            schema=scheme_prompt,
            evidence=evidence,
            semantic_model="",
            plan_validation_error=plan_validation_error,
            format=format_instruction,
        )

        logger.info(f"Planner prompt rendering complete for query: '{rewrite_query}'")

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )

        content = resp.choices[0].message.content or "{}"
        clean_content = content.strip()
        if clean_content.startswith("```"):
            clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
            clean_content = re.sub(r"\s*```$", "", clean_content)

        # 校验 Plan JSON 结构合法性
        try:
            plan_obj = Plan.model_validate_json(clean_content)
            plan_json_str = plan_obj.model_dump_json()
        except Exception as e:
            logger.warning(f"Failed to validate plan model from LLM: {clean_content}, error: {e}")
            plan_json_str = clean_content

        logger.info(f"Planner node generated plan:\n{plan_json_str}")

        return {
            StateKey.PLAN.value: plan_json_str
        }


planner_node = PlannerNode()

