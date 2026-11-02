from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ToolParameters(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    instruction: Optional[str] = Field(
        default=None,
        description="当 tool_to_use 是 SQL_GENERATE_NODE 时填详细 SQL 需求；当是 PYTHON_GENERATE_NODE 时填详细编程需求",
    )
    summary_and_recommendations: Optional[str] = Field(
        default=None,
        description="REPORT_GENERATOR_NODE 专用，报告大纲、需要回答的关键问题和建议方向",
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="运行态字段：SQL_GENERATE_NODE 运行后填入实际生成的 SQL",
    )


class ExecutionStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    step: int = Field(..., description="步骤顺序号，从 1 开始")
    tool_to_use: str = Field(
        ...,
        description="工具名称：SQL_GENERATE_NODE / PYTHON_GENERATE_NODE / REPORT_GENERATOR_NODE",
    )
    tool_parameters: ToolParameters = Field(..., description="工具参数")


class Plan(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thought_process: str = Field(..., description="简要描述分析思路，必须明确提到检查了哪些表和字段")
    execution_plan: list[ExecutionStep] = Field(..., description="执行计划的步骤列表")

