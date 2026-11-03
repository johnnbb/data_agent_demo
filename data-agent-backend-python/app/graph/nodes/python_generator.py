import json
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


def extract_raw_python(content: str) -> str:
    """提取纯净的 Python 代码，剥离任何 Markdown 代码块标签与多余包裹"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class PythonGeneratorNode:
    """
    Python 分析代码生成节点。
    根据表结构、前序 SQL 查询结果样例、当前步骤指令与计算参数，
    生成可直接在沙箱中由 sys.stdin 读取 JSON 数据并执行分析的 Python 脚本。
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
        current_step = state.get(StateKey.CURRENT_STEP.value, 1) or 1
        logger.info(f"[PythonGeneratorNode] Generating Python script for Step {current_step}")

        # 1. 提取表结构 Schema Prompt
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

        # 2. 提取输入数据样例（优先从 SQL_EXECUTION_RESULT 提取，兼容 EXECUTION_OUTPUT）
        sample_data = []
        sql_exec_result = state.get(StateKey.SQL_EXECUTION_RESULT.value)
        if isinstance(sql_exec_result, dict):
            sample_data = sql_exec_result.get("resultSet", {}).get("data", [])
        elif not sample_data:
            exec_output = state.get(StateKey.EXECUTION_OUTPUT.value)
            if isinstance(exec_output, dict):
                # 寻找最近的 step 产出
                for key in sorted(exec_output.keys(), reverse=True):
                    val = exec_output[key]
                    if isinstance(val, dict) and "data" in val:
                        sample_data = val.get("data", [])
                        break

        # 如果数据量大，截取前 10 行作为 prompt 样例，防止超大 token
        sample_input_prompt = sample_data[:10] if isinstance(sample_data, list) else sample_data
        sample_input_json = json.dumps(sample_input_prompt, ensure_ascii=False)

        # 3. 提取当前步骤方案指导与参数
        plan_raw = state.get(StateKey.PLAN.value)
        tool_params: dict[str, Any] = {}
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
                    tool_params = step.tool_parameters.model_dump(exclude_none=True)
            except Exception as e:
                logger.warning(f"[PythonGeneratorNode] Failed to extract plan step parameters: {e}")

        plan_desc_json = json.dumps(tool_params, ensure_ascii=False) if tool_params else "{}"

        # 4. 渲染 python-generator.st 提示词
        rendered_prompt = prompt_manager.render_python_generator(
            python_memory="500",
            python_timeout="500",
            database_schema=scheme_prompt,
            sample_input=sample_input_json,
            plan_description=plan_desc_json,
        )

        # 5. 调用大模型生成代码
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": rendered_prompt},
                {"role": "user", "content": "请根据要求编写纯净可执行的 Python 脚本，严禁使用 markdown 标记包裹。"},
            ],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        raw_code = resp.choices[0].message.content or ""
        clean_code = extract_raw_python(raw_code)
        logger.info(f"[PythonGeneratorNode] Generated Python code:\n{clean_code}")

        return {
            StateKey.PYTHON_GENERATION_RESULT.value: clean_code,
        }


python_generator_node = PythonGeneratorNode()

