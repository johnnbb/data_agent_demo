import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

import sqlglot
from sqlglot import exp

from app.graph.state import AgentState
from app.graph.spec import StateKey, NodeName
from app.graph.plan_models import Plan
from app.graph.sql_models import SqlResultSet, DisplaySpec, SqlExecutionResult

logger = logging.getLogger(__name__)


def validate_sql_ast(sql: str, dialect: str = "sqlite") -> tuple[bool, Optional[str]]:
    """使用 sqlglot 进行 AST 静态语法树分析与只读安全性校验（拦截 DROP, DELETE, UPDATE, INSERT, ALTER 等）"""
    clean_sql = sql.strip().rstrip(";")
    if not clean_sql:
        return False, "SQL 语句为空"
    try:
        parsed = sqlglot.parse(clean_sql, read=dialect)
        for stmt in parsed:
            if stmt is None:
                continue
            # 1. 深度拦截任何破坏性或写入 DDL/DML 表达式
            forbidden = (
                exp.Drop,
                exp.Delete,
                exp.Update,
                exp.Insert,
                exp.Alter,
                exp.Command,
            )
            for f in forbidden:
                if stmt.find(f) or isinstance(stmt, f):
                    return False, f"安全校验拦截：只允许执行只读 SELECT 查询，检测到破坏性语句: {stmt.key.upper()}"
            # 2. 确保顶级或核心结构属于 Select / Union
            if not (isinstance(stmt, (exp.Select, exp.Union)) or stmt.find(exp.Select)):
                return False, f"安全校验拦截：只允许执行只读 SELECT 查询，不支持操作: {stmt.key.upper()}"
        return True, None
    except Exception as e:
        return False, f"SQL 语法解析失败 (AST 校验未通过): {e}"


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
    通过 AST 只读校验与真实数据库物理只读连接执行 SQL 语句，
    并在遇到错误时联动 SQL_GENERATION 触发自纠错重试机制（最多 3 次）。
    """

    def __call__(self, state: AgentState) -> dict[str, Any]:
        sql = state.get(StateKey.SQL_GENERATION_RESULT.value, "") or ""
        database_id = state.get(StateKey.DATABASE_ID.value, "california_schools") or "california_schools"
        current_step = state.get(StateKey.CURRENT_STEP.value, 1) or 1
        repair_count = state.get(StateKey.SQL_REPAIR_COUNT.value, 0) or 0

        logger.info(
            f"[SqlExecuteNode] Executing SQL on db '{database_id}' for Step {current_step} "
            f"(Repair attempt: {repair_count}/3):\n{sql}"
        )

        db_path = locate_sqlite_database(database_id)
        if not db_path:
            err_msg = f"Database file for '{database_id}' not found in resources."
            logger.error(f"[SqlExecuteNode] {err_msg}")
            result_set = SqlResultSet(column=[], data=[], errorMsg=err_msg)
            display = DisplaySpec(type="table", title="数据库连接失败", x=None, y=[])
            result = SqlExecutionResult(resultSet=result_set, display=display)
            return {
                StateKey.CURRENT_STEP.value: current_step + 1,
                StateKey.SQL_REPAIR_COUNT.value: 0,
                StateKey.SQL_ERROR_MESSAGE.value: None,
                StateKey.SQL_NEXT_NODE.value: NodeName.PLAN_EXECUTION.value,
                StateKey.SQL_EXECUTION_RESULT.value: result.model_dump(),
                StateKey.EXECUTION_OUTPUT.value: {f"step_{current_step}": result_set.model_dump()},
            }

        error_msg: Optional[str] = None
        clean_columns: list[str] = []
        raw_data: list[dict[str, Any]] = []

        # 1. 前置 AST 静态语法与只读校验
        is_valid, validation_err = validate_sql_ast(sql, dialect="sqlite")
        if not is_valid:
            error_msg = validation_err
            logger.warning(f"[SqlExecuteNode] AST validation rejected SQL: {error_msg}")
        else:
            # 2. 建立物理只读连接（mode=ro 保护机制）
            try:
                db_uri = f"file:{db_path.resolve()}?mode=ro"
                conn = sqlite3.connect(db_uri, uri=True)
                cursor = conn.cursor()
                cursor.execute(sql)

                raw_columns = [desc[0] for desc in cursor.description] if cursor.description else []
                clean_columns = [clean_name(col) for col in raw_columns]

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
            except Exception as e:
                logger.error(f"[SqlExecuteNode] SQLite execution error: {e}")
                error_msg = str(e)

        # 3. 错误自纠错路由判断
        if error_msg is not None:
            result_set = SqlResultSet(column=[], data=[], errorMsg=error_msg)
            display = DisplaySpec(type="table", title=f"SQL执行失败: {error_msg}", x=None, y=[])
            result = SqlExecutionResult(resultSet=result_set, display=display)

            # 未超限：回跳 SQL_GENERATION 触发自纠错
            if repair_count < 3:
                logger.warning(
                    f"[SqlExecuteNode] SQL error on Step {current_step} (Retry {repair_count + 1}/3): {error_msg}. "
                    f"Routing back to SQL_GENERATION for self-reflection."
                )
                return {
                    StateKey.SQL_REPAIR_COUNT.value: repair_count + 1,
                    StateKey.SQL_ERROR_MESSAGE.value: error_msg,
                    StateKey.SQL_NEXT_NODE.value: NodeName.SQL_GENERATION.value,
                    StateKey.SQL_EXECUTION_RESULT.value: result.model_dump(),
                }
            else:
                logger.error(
                    f"[SqlExecuteNode] Max SQL repair retries (3) exhausted for Step {current_step}. "
                    f"Proceeding with error."
                )
        else:
            # 正常执行成功
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
            result_set = SqlResultSet(column=clean_columns, data=raw_data, errorMsg=None)
            result = SqlExecutionResult(resultSet=result_set, display=display)

        # 4. 回填实际执行的 SQL 到当前步骤参数中
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

        # 5. 安全合并多步骤历史输出
        existing_output = state.get(StateKey.EXECUTION_OUTPUT.value) or {}
        if not isinstance(existing_output, dict):
            existing_output = {}

        merged_output = {
            **existing_output,
            f"step_{current_step}": result_set.model_dump(),
        }

        ret = {
            StateKey.CURRENT_STEP.value: current_step + 1,
            StateKey.SQL_REPAIR_COUNT.value: 0,
            StateKey.SQL_ERROR_MESSAGE.value: None,
            StateKey.SQL_NEXT_NODE.value: NodeName.PLAN_EXECUTION.value,
            StateKey.SQL_EXECUTION_RESULT.value: result.model_dump(),
            StateKey.EXECUTION_OUTPUT.value: merged_output,
        }
        if updated_plan is not None:
            ret[StateKey.PLAN.value] = updated_plan
        return ret


sql_execute_node = SqlExecuteNode()

