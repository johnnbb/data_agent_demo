import json
import logging
from typing import Any, Optional
from openai import OpenAI

from app.config import settings
from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.prompts.manager import prompt_manager

logger = logging.getLogger(__name__)


class PythonAnalyzeNode:
    """
    Python 分析解读节点。
    接收 Python 沙箱脚本执行的结构化产物，结合用户原始查询与重写意图，
    利用大模型将指标与计算结果提炼为清晰易读的自然语言分析总结，
    回填至 EXECUTION_OUTPUT['step_{step}_analysis'] 并递增步骤计数器。
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
        rewrite_query = state.get(StateKey.REWRITE_QUERY.value) or state.get("input", "")
        python_res = state.get(StateKey.PYTHON_EXECUTION_RESULT.value)

        # 格式化 python_output 文本供提示词分析
        if isinstance(python_res, dict):
            if python_res.get("success", False):
                raw_out = python_res.get("output", "")
                python_output_str = raw_out if raw_out else "【脚本执行成功，无额外标准输出】"
            else:
                raw_err = python_res.get("error", "")
                python_output_str = f"分析过程中出现错误：\n{raw_err}" if raw_err else "分析过程中出现错误。"
        elif isinstance(python_res, str):
            python_output_str = python_res
        else:
            python_output_str = "【无分析输出】"

        logger.info(f"[PythonAnalyzeNode] Analyzing step {current_step} output for query: '{rewrite_query}'")

        # 渲染 python-analyze.st
        prompt_text = prompt_manager.render_python_analyze(
            python_output=python_output_str,
            user_query=rewrite_query,
        )

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
            extra_body={"enable_thinking": False},
        )

        analysis_text = resp.choices[0].message.content or ""
        logger.info(f"[PythonAnalyzeNode] Step {current_step} analysis result:\n{analysis_text}")

        # 累加保存到 EXECUTION_OUTPUT
        existing_output = state.get(StateKey.EXECUTION_OUTPUT.value)
        if isinstance(existing_output, dict):
            new_output = dict(existing_output)
        elif isinstance(existing_output, str):
            try:
                new_output = json.loads(existing_output)
            except Exception:
                new_output = {"previous_output": existing_output}
        else:
            new_output = {}

        new_output[f"step_{current_step}_analysis"] = analysis_text

        return {
            StateKey.EXECUTION_OUTPUT.value: new_output,
            StateKey.CURRENT_STEP.value: current_step + 1,
        }


python_analyze_node = PythonAnalyzeNode()

