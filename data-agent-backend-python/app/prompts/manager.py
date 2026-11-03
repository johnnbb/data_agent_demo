from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).parent


class PromptManager:
    """
    提示词模板管理器。
    负责安全读取 app/prompts/*.st 模板并提供参数渲染能力。
    """

    def __init__(self, prompts_dir: Path = PROMPTS_DIR):
        self.prompts_dir = prompts_dir

    def _read_template(self, filename: str) -> str:
        filepath = self.prompts_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt template file not found: {filepath}")
        return filepath.read_text(encoding="utf-8")

    def render(self, filename: str, **kwargs: Any) -> str:
        """
        加载并渲染指定的 .st 模板。
        使用安全替换逻辑，防止模板内含有的非占位符花括号（如示例 JSON 里的 {}）报错。
        渲染完成后将 StringTemplate 转义的 \\{ 和 \\} 还原为 { 和 }。
        """
        template = self._read_template(filename)
        rendered = template
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            rendered = rendered.replace(placeholder, str(value))
        rendered = rendered.replace(r"\{", "{").replace(r"\}", "}")
        return rendered

    # 快捷方法
    def render_evidence_query_rewrite(self, latest_query: str, format_str: str, multi_turn: str = "(无)") -> str:
        return self.render(
            "evidence-query-rewrite.st",
            latest_query=latest_query,
            format=format_str,
            multi_turn=multi_turn,
        )

    def render_business_knowledge(self, business_knowledge: str) -> str:
        return self.render("business-knowledge.st", businessKnowledge=business_knowledge if business_knowledge else "无")

    def render_agent_knowledge(self, agent_knowledge: str) -> str:
        return self.render("agent-knowledge.st", agentKnowledge=agent_knowledge if agent_knowledge else "无")

    def render_python_generator(
        self,
        python_memory: str = "500",
        python_timeout: str = "500",
        database_schema: str = "",
        sample_input: str = "",
        plan_description: str = "",
    ) -> str:
        return self.render(
            "python-generator.st",
            python_memory=python_memory,
            python_timeout=python_timeout,
            database_schema=database_schema,
            sample_input=sample_input,
            plan_description=plan_description,
        )

    def render_python_analyze(self, python_output: Any, user_query: str) -> str:
        return self.render(
            "python-analyze.st",
            python_output=python_output,
            user_query=user_query,
        )

    def render_report_generator(
        self,
        user_requirements_and_plan: str,
        analysis_steps_and_data: str,
        summary_and_recommendations: str,
        json_example: str,
        optimization_section: str = "",
    ) -> str:
        return self.render(
            "report-generator-plain.st",
            user_requirements_and_plan=user_requirements_and_plan,
            analysis_steps_and_data=analysis_steps_and_data,
            summary_and_recommendations=summary_and_recommendations,
            json_example=json_example,
            optimization_section=optimization_section,
        )


prompt_manager = PromptManager()

