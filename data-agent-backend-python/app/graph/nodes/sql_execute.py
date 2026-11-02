import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.graph.state import AgentState
from app.graph.spec import StateKey
from app.graph.plan_models import Plan
from app.graph.sql_models import SqlResultSet, DisplaySpec, SqlExecutionResult

logger = logging.getLogger(__name__)


def locate_sqlite_database(database_id: str) -> Optional[Path]:
    """定位 BIRD Benchmark 的 SQLite 数据库文件（使用 Python 后端内聚数据目录）"""
    candidates = [
        Path(__file__).resolve().parents[3] / "data" / "dev_20240627" / "dev_databases" / database_id / f"{database_id}.sqlite",
        Path.cwd() / "data-agent-backend-python" / "data" / "dev_20240627" / "dev_databases" / database_id / f"{database_id}.sqlite",
        Path.cwd() / "data" / "dev_20240627" / "dev_databases" / database_id / f"{database_id}.sqlite",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def clean_name(name: str) -> str:
    return name.replace("`", "").replace('"', "")


class SqlExecuteNode:
    """
    SQL 数据库执行节点（对应 Kotlin 的 SqlExecuteNode.kt 与 ResultSetBuilder.kt）。
    通过真实数据库连接执行 SQL 语句，提取表头与行数据（1000 行安全截断），
    构造 DisplaySpec 可视化元数据，并推进步骤指针（currentStep + 1）。
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        sql = state.get(StateKey.SQL_GENERATION_RESULT.value, "") or ""
        database_id = state.get(StateKey.DATABASE_ID.value, "california_schools") or "california_schools"
        current_step = state.get(StateKey.CURRENT_STEP.value, 1) or 1

        logger.info(f"[SqlExecuteNode] Executing SQL on db '{database_id}' for Step {current_step}:\n{sql}")

        db_path = locate_sqlite_database(database_id)
        if not db_path:
            err_msg = f"Database file for '{database_id}' not found in resources."
            logger.error(f"[SqlExecuteNode] {err_msg}")
            result_set = SqlResultSet(column=[], data=[], errorMsg=err_msg)
            display = DisplaySpec(type="table", title="数据库连接失败", x=None, y=[])
            result = SqlExecutionResult(resultSet=result_set, display=display)
            return {
                StateKey.CURRENT_STEP.value: current_step + 1,
                StateKey.SQL_EXECUTION_RESULT.value: result.model_dump(),
                StateKey.EXECUTION_OUTPUT.value: {f"step_{current_step}": result_set.model_dump()},
            }

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(sql)

            # 1. 提取列名并清洗引号
            raw_columns = [desc[0] for desc in cursor.description] if cursor.description else []
            clean_columns = [clean_name(col) for col in raw_columns]

            # 2. 提取数据行并限制 1000 行
            raw_data: list[dict[str, Any]] = []
            count = 0
            for row in cursor:
                if count >= 1000:
                    break
                row_dict = {}
                for idx, col_name in enumerate(clean_columns):
                    val = row[idx]
                    row_dict[col_name] = "" if val is None else str(val)
                raw_data.append(row_dict)
                count += 1

            cursor.close()
            conn.close()

            logger.info(
                f"[SqlExecuteNode] SQL executed successfully: {len(clean_columns)} columns, {len(raw_data)} rows returned"
            )

            result_set = SqlResultSet(
                column=clean_columns,
                data=raw_data,
                errorMsg=None,
            )

            # 3. 构造 DisplaySpec 可视化元数据
            if not clean_columns:
                display = DisplaySpec(type="table", title="SQL执行成功，无数据返回", x=None, y=[])
            else:
                x_axis = clean_columns[0]
                y_axis = clean_columns[1:] if len(clean_columns) > 1 else []
                display = DisplaySpec(
                    type="table",
                    title="SQL执行结果",
                    x=x_axis,
                    y=y_axis,
                )

            result = SqlExecutionResult(resultSet=result_set, display=display)

        except Exception as e:
            logger.error(f"[SqlExecuteNode] SQL execution error: {e}")
            result_set = SqlResultSet(column=[], data=[], errorMsg=str(e))
            display = DisplaySpec(type="table", title=f"SQL执行失败: {e}", x=None, y=[])
            result = SqlExecutionResult(resultSet=result_set, display=display)

        # 4. 回填实际执行的 SQL 到当前步骤参数中（方便后续 Report 节点可追溯渲染）
        plan_raw = state.get(StateKey.PLAN.value)
        updated_plan = None
        if plan_raw:
            try:
                if isinstance(plan_raw, Plan):
                    plan_obj = plan_raw.model_copy(deep=True)
                elif isinstance(plan_raw, dict):
                    plan_obj = Plan.model_validate(plan_raw)
                else:
                    plan_obj = Plan.model_validate_json(str(plan_raw))

                if 1 <= current_step <= len(plan_obj.execution_plan):
                    plan_obj.execution_plan[current_step - 1].tool_parameters.sql_query = sql
                    updated_plan = plan_obj.model_dump()
            except Exception as e:
                logger.warning(f"[SqlExecuteNode] Failed to stamp sql_query into plan: {e}")

        # 5. 安全合并多步骤历史输出，防止后续步骤覆盖前置步骤产物
        existing_output = state.get(StateKey.EXECUTION_OUTPUT.value) or {}
        if not isinstance(existing_output, dict):
            existing_output = {}

        merged_output = {
            **existing_output,
            f"step_{current_step}": result_set.model_dump(),
        }

        ret = {
            StateKey.CURRENT_STEP.value: current_step + 1,
            StateKey.SQL_EXECUTION_RESULT.value: result.model_dump(),
            StateKey.EXECUTION_OUTPUT.value: merged_output,
        }
        if updated_plan is not None:
            ret[StateKey.PLAN.value] = updated_plan
        return ret


sql_execute_node = SqlExecuteNode()

