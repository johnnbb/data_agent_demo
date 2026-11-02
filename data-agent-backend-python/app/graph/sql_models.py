from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class SqlResultSet(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    column: list[str] = Field(default_factory=list, description="列名列表/表头")
    data: list[dict[str, Any]] = Field(default_factory=list, description="行数据列表，每行为列名到单元格值的映射")
    errorMsg: Optional[str] = Field(default=None, description="SQL执行异常信息，若成功为None")


class DisplaySpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    type: str = Field(default="table", description="图表或展示类型，如 table, bar, line 等")
    title: Optional[str] = Field(default="SQL执行结果", description="图表或结果标题")
    x: Optional[str] = Field(default=None, description="X轴字段名或分类维度列")
    y: list[str] = Field(default_factory=list, description="Y轴字段名列表或度量指标列")


class SqlExecutionResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resultSet: SqlResultSet = Field(..., description="数据结果集")
    display: DisplaySpec = Field(..., description="前端可视化规格")

