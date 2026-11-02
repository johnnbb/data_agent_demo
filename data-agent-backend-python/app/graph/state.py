from typing import TypedDict, Optional, Any


class AgentState(TypedDict, total=False):
    # 核心数据输入
    input: str
    databaseId: str
    MULTI_TURN_CONTEXT: Optional[str]

    # Evidence Recall 阶段产物（与前端大写常量对齐）
    REWRITE_QUERY: str
    EVIDENCE: str

    # Scheme Recall 阶段产物（与前端大写常量对齐）
    TABLE_SCHEME: list[dict[str, Any]]
    COLUMN_SCHEME: list[dict[str, Any]]

    # Table Relation 阶段产物（与前端大写常量对齐）
    TABLE_RELATION_OUTPUT: dict[str, Any]

    # 可行性研判与任务规划产物（与前端大写常量对齐）
    FEASIBILITY_ASSESSMENT_NODE_OUTPUT: str
    PLANNER_NODE_OUTPUT: str | dict[str, Any]
    PLAN_REPAIR_COUNT: Optional[int]
    PLAN_NEXT_NODE: Optional[str]
    PLAN_CURRENT_STEP: Optional[int]
    PLAN_EXECUTE_NODE_OUTPUT: Optional[str | dict[str, Any]]
    PLAN_VALIDATION_ERROR: Optional[str]

    # 人工审核交互状态（与 A2A Metadata 及 HumanFeedbackNode 对齐）
    confirmationApproved: Optional[bool]
    confirmationFeedback: Optional[str]
    HUMAN_NEXT_NODE: Optional[str]

    # SQL 生成与执行阶段产物（与前端大写常量对齐）
    SQL_GENERATE_OUTPUT: Optional[str]
    SQL_EXECUTE_OUTPUT: Optional[dict[str, Any]]

    # Python 生成、执行与分析产物（与前端大写常量对齐）
    PYTHON_GENERATE_NODE_OUTPUT: Optional[str]
    PYTHON_EXECUTE_NODE_OUTPUT: Optional[dict[str, Any]]

    # 最终业务报告产物（与前端大写常量对齐）
    REPORT_GENERATOR_NODE_OUTPUT: Optional[str]
