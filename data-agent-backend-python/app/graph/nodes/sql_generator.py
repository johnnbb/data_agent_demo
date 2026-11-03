import logging
import re
from typing import Any, Optional
from openai import OpenAI

from app.config import settings
from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.graph.plan_models import Plan
from app.retrieval.schema_models import Schema
from app.prompts.manager import prompt_manager

logger = logging.getLogger(__name__)


def extract_raw_sql(content: str) -> str:
    """提取纯净的 SQL 语句，去除可能存在的 Markdown 代码块标记与注释"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    cleaned = text.strip()
    # 统一确保以分号结尾
    if cleaned and not cleaned.endswith(";"):
        cleaned += ";"
    return cleaned


class SqlGeneratorNode:
    """
    SQL 代码生成与反思修复节点。
    根据当前执行步骤的指令（instruction），结合高保真精选 Schema 和领域知识，
    生成方言匹配的精准 SQL 查询语句。
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
        dialect: str = "sqlite",
    ):
        self.chat_model = chat_model or settings.CHAT_MODEL
        self.dialect = dialect
        self.client = OpenAI(
            api_key=openai_api_key or settings.OPENAI_API_KEY,
            base_url=openai_base_url or settings.OPENAI_BASE_URL,
        )

    def __call__(self, state: AgentState) -> dict[str, Any]:
        # 1. 提取当前步骤的特定指令
        current_step = state.get(StateKey.CURRENT_STEP.value, 1) or 1
        plan_raw = state.get(StateKey.PLAN.value)

        instruction = ""
        if plan_raw:
            try:
                if isinstance(plan_raw, Plan):
                    plan_obj = plan_raw
                elif isinstance(plan_raw, dict):
                    plan_obj = Plan.model_validate(plan_raw)
                else:
                    plan_obj = Plan.model_validate_json(str(plan_raw))

                if 1 <= current_step <= len(plan_obj.execution_plan):
                    step = plan_obj.execution_plan[current_step - 1]
                    instruction = step.tool_parameters.instruction or ""
            except Exception as e:
                logger.warning(f"[SqlGeneratorNode] Failed to extract instruction from plan: {e}")

        if not instruction:
            instruction = state.get(StateKey.REWRITE_QUERY.value) or state.get("input", "")

        # 2. 提取 Schema 与背景知识
        schema_data = state.get(StateKey.TABLE_RELATION.value)
        if isinstance(schema_data, dict):
            schema_obj = Schema.model_validate(schema_data)
            scheme_prompt = schema_obj.build_scheme_prompt()
        elif isinstance(schema_data, str):
            try:
                schema_obj = Schema.model_validate_json(schema_data)
                scheme_prompt = schema_obj.build_scheme_prompt()
            except Exception:
                scheme_prompt = schema_data
        else:
            scheme_prompt = "【无表结构】"

        rewrite_query = state.get(StateKey.REWRITE_QUERY.value) or state.get("input", "")
        evidence = state.get(StateKey.EVIDENCE.value, "")

        # 3. 检查是否有执行报错需要反思重修（自纠错模式 vs 首次生成模式）
        sql_error_msg = state.get(StateKey.SQL_ERROR_MESSAGE.value)
        failed_sql = state.get(StateKey.SQL_GENERATION_RESULT.value, "")

        if sql_error_msg:
            logger.info(
                f"[SqlGeneratorNode] Step {current_step} triggering self-reflection repair for error: {sql_error_msg}"
            )
            sql_prompt = prompt_manager.render(
                "sql-error-fixer.st",
                dialect=self.dialect,
                error_message=sql_error_msg,
                schema_info=scheme_prompt,
                execution_description=instruction,
                error_sql=failed_sql,
                question=rewrite_query,
                evidence=evidence,
            )
            user_prompt = "请针对报错信息和 Schema 仔细分析，仅输出修复后的纯 SQL 查询语句，严禁包含任何解释或 Markdown 代码块。"
        else:
            # 首次标准生成
            sql_prompt = prompt_manager.render(
                "new-sql-generate.st",
                dialect=self.dialect,
                question=rewrite_query,
                schema_info=scheme_prompt,
                evidence=evidence,
                semantic_lookup_rules="(无)",
                semantic_enum_rules="(无)",
                execution_description=instruction,
            )
            user_prompt = "请只输出纯 SQL 查询语句，不要包含任何解释或额外文字。"

        logger.info(
            f"[SqlGeneratorNode] Generating SQL for Step {current_step}: '{instruction}' (Repair mode: {bool(sql_error_msg)})"
        )

        # 4. 调用大模型生成 SQL
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": sql_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        raw_sql = resp.choices[0].message.content or ""
        clean_sql = extract_raw_sql(raw_sql)
        logger.info(f"[SqlGeneratorNode] Generated SQL:\n{clean_sql}")

        return {
            StateKey.SQL_GENERATION_RESULT.value: clean_sql,
        }


sql_generator_node = SqlGeneratorNode()

