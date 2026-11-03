import json
import logging
from typing import Any, Optional
from openai import OpenAI

from app.config import settings
from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.graph.plan_models import Plan
from app.prompts.manager import prompt_manager

logger = logging.getLogger(__name__)


class ReportGeneratorNode:
    """
    终极综合分析报告生成节点。
    消费全流程历史沉淀的执行数据（SQL 查询数据、实际执行 SQL、Python 分析解释），
    结合用户原始诉求与规划思路，驱动 LLM 汇总生成结构化 Markdown 报告与 ECharts 图表，
    达到可执行、可解释、可追溯、可交付的完整闭环。
    """

    clean_json_example = """{
    "title": { "text": "月度销售额" },
    "tooltip": { "trigger": "axis" },
    "xAxis": { "type": "category", "data": ["1月", "2月"] },
    "yAxis": { "type": "value" },
    "series": [
        { "type": "bar", "data": [120, 200] }
    ]
}"""

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
        logger.info("[ReportGeneratorNode] Applying ReportGeneratorNode...")
        rewrite_query = state.get(StateKey.REWRITE_QUERY.value) or state.get("input", "")

        # 1. 解析全局规划 Plan
        plan_raw = state.get(StateKey.PLAN.value)
        if isinstance(plan_raw, Plan):
            plan_obj = plan_raw
        elif isinstance(plan_raw, dict):
            plan_obj = Plan.model_validate(plan_raw)
        else:
            plan_obj = Plan.model_validate_json(str(plan_raw))

        # 2. 提取执行产物
        execution_results = state.get(StateKey.EXECUTION_OUTPUT.value) or {}
        if not isinstance(execution_results, dict):
            try:
                execution_results = json.loads(str(execution_results))
            except Exception:
                execution_results = {"output": str(execution_results)}

        # 3. 提取当前报告步骤设定的总结与建议要求
        current_step = state.get(StateKey.CURRENT_STEP.value, 1) or 1
        summary_and_recommendations = ""
        if 1 <= current_step <= len(plan_obj.execution_plan):
            current_step_obj = plan_obj.execution_plan[current_step - 1]
            if current_step_obj.tool_parameters:
                summary_and_recommendations = current_step_obj.tool_parameters.summary_and_recommendations or ""

        # 4. 拼装两大核心切片
        user_requirements_and_plan = self.build_user_requirements_and_plan(rewrite_query, plan_obj)
        analysis_steps_and_data = self.build_analysis_steps_and_data(plan_obj, execution_results)

        # 5. 渲染报告生成提示词
        report_prompt = prompt_manager.render_report_generator(
            user_requirements_and_plan=user_requirements_and_plan,
            analysis_steps_and_data=analysis_steps_and_data,
            summary_and_recommendations=summary_and_recommendations,
            json_example=self.clean_json_example,
            optimization_section="",
        )

        logger.info(f"[ReportGeneratorNode] Prompt rendered. Prompt length: {len(report_prompt)}")

        # 6. 调用大模型生成 Markdown 报告
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "user", "content": report_prompt},
            ],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        report_text = resp.choices[0].message.content or ""
        logger.info(f"[ReportGeneratorNode] Report generation complete. Output length: {len(report_text)}")

        return {
            StateKey.REPORT_RESULT.value: report_text,
        }

    @classmethod
    def build_user_requirements_and_plan(cls, user_input: str, plan: Plan) -> str:
        """构建切片一：用户原始需求、思考过程及执行计划概述"""
        lines = [
            "## 用户原始需求",
            user_input,
            "",
            "## 执行计划概述",
            f"**思考过程**: {plan.thought_process}",
            "",
            "## 详细执行步骤",
        ]
        for index, step in enumerate(plan.execution_plan):
            lines.append(f"### 步骤 {index + 1}: 步骤编号 {step.step}")
            lines.append(f"**工具**: {step.tool_to_use}")
            if step.tool_parameters and step.tool_parameters.instruction:
                lines.append(f"**参数描述**: {step.tool_parameters.instruction}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def build_analysis_steps_and_data(cls, plan: Plan, execution_results: dict[str, Any]) -> str:
        """构建切片二：真实执行步骤、数据结果及关联的 Python 分析"""
        lines = ["## 数据执行结果"]

        if not execution_results:
            lines.append("暂无执行结果数据")
            return "\n".join(lines)

        execution_plan = plan.execution_plan if plan and plan.execution_plan else []

        for step_key, step_result in execution_results.items():
            # 过滤掉以 _analysis 结尾的 key
            if step_key.endswith("_analysis"):
                continue

            lines.append(f"### {step_key}")

            # 尝试获取对应的步骤描述与运行态 SQL
            try:
                step_index = int(step_key.replace("step_", "")) - 1
                if 0 <= step_index < len(execution_plan):
                    step = execution_plan[step_index]
                    lines.append(f"**步骤编号**: {step.step}")
                    lines.append(f"**使用工具**: {step.tool_to_use}")

                    if step.tool_parameters:
                        if step.tool_parameters.instruction:
                            lines.append(f"**参数描述**: {step.tool_parameters.instruction}")
                        if step.tool_parameters.sql_query:
                            lines.append(f"**执行SQL**: \n```sql\n{step.tool_parameters.sql_query}\n```")
            except (ValueError, TypeError):
                pass

            if isinstance(step_result, dict) and "data" in step_result and isinstance(step_result["data"], list):
                total_len = len(step_result["data"])
                if total_len > 20:
                    sample_dict = {
                        "column": step_result.get("column", []),
                        "data": step_result["data"][:20],
                        "total_rows": total_len,
                        "note": f"数据量较大，共 {total_len} 行，此处仅展示前 20 行样例",
                    }
                    formatted_result = json.dumps(sample_dict, ensure_ascii=False, indent=2)
                else:
                    formatted_result = json.dumps(step_result, ensure_ascii=False, indent=2)
            elif isinstance(step_result, (dict, list)):
                formatted_result = json.dumps(step_result, ensure_ascii=False, indent=2)
            else:
                formatted_result = str(step_result)

            lines.append(f"**执行结果**: \n```json\n{formatted_result}\n```\n")

            analysis_key = f"{step_key}_analysis"
            if analysis_key in execution_results and execution_results[analysis_key]:
                lines.append(f"**Python 分析结果**: {execution_results[analysis_key]}\n")

        return "\n".join(lines)


report_generator_node = ReportGeneratorNode()

