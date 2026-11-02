import logging
from typing import Any, Optional
from openai import OpenAI

from app.config import settings
from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.prompts.manager import prompt_manager
from app.retrieval.schema_models import Schema

logger = logging.getLogger(__name__)


class FeasibilityAssessmentNode:
    """
    可行性研判门禁节点（对应 Kotlin 的 FeasibilityAssessmentNode.kt）。
    结合规范化查询、业务 Evidence 和提取后的局部 Schema，研判用户问题是否可回答。
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
        multi_turn = state.get(StateKey.MULTI_TURN_CONTEXT.value) or "(无)"
        schema_data = state.get(StateKey.TABLE_RELATION.value)

        # 还原 Schema 并生成提示词所需的 DDL 表达
        if isinstance(schema_data, dict):
            schema_obj = Schema.model_validate(schema_data)
            recalled_schema = schema_obj.build_scheme_prompt()
        elif isinstance(schema_data, str):
            recalled_schema = schema_data
        else:
            recalled_schema = "【暂无相关表结构】"

        prompt = prompt_manager.render(
            "feasibility-assessment.st",
            recalled_schema=recalled_schema,
            evidence=evidence,
            canonical_query=rewrite_query,
            multi_turn=multi_turn,
        )

        logger.info(f"Feasibility assessment prompt for query: '{rewrite_query}'")

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        feasibility_result = resp.choices[0].message.content or ""
        logger.info(f"Feasibility assessment result:\n{feasibility_result}")

        return {
            StateKey.FEASIBILITY_RESULT.value: feasibility_result
        }


feasibility_assessment_node = FeasibilityAssessmentNode()

