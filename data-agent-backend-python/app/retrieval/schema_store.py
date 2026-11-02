import logging
from typing import Optional
import psycopg
from app.config import settings
from app.retrieval.schema_models import (
    DbColumnSchemaView,
    DbTableSchemaView,
    DbForeignKeySchemaView,
    DbTableSimpleView,
)

logger = logging.getLogger(__name__)


class SchemaStore:
    """
    数据资产元数据存储仓库（对应 Kotlin 的 DbForeignKeyRepository 和 DbTableRepository）。
    负责从 PostgreSQL 关系表中查询结构、外键与字段详情。
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.DATABASE_URL

    def find_foreign_keys_by_database_id(self, database_id: str) -> list[DbForeignKeySchemaView]:
        """查询指定数据库的所有外键关系"""
        sql = """
            SELECT
                fk.id as fk_id,
                sc.id as sc_id, sc.name as sc_name, sc.type as sc_type, sc.description as sc_desc, sc.is_primary_key as sc_pk,
                st.id as st_id, st.name as st_name,
                tc.id as tc_id, tc.name as tc_name, tc.type as tc_type, tc.description as tc_desc, tc.is_primary_key as tc_pk,
                tt.id as tt_id, tt.name as tt_name
            FROM db_foreign_key fk
            JOIN db_column sc ON fk.source_column_id = sc.id
            JOIN db_table st ON sc.table_id = st.id
            JOIN db_column tc ON fk.target_column_id = tc.id
            JOIN db_table tt ON tc.table_id = tt.id
            WHERE st.database_id = %(database_id)s
        """
        results: list[DbForeignKeySchemaView] = []
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"database_id": database_id})
                rows = cur.fetchall()
                for r in rows:
                    source_col = DbColumnSchemaView(
                        id=str(r[1]),
                        name=r[2],
                        type=r[3],
                        description=r[4],
                        is_primary_key=bool(r[5]),
                        db_table=DbTableSimpleView(id=str(r[6]), name=r[7]),
                    )
                    target_col = DbColumnSchemaView(
                        id=str(r[8]),
                        name=r[9],
                        type=r[10],
                        description=r[11],
                        is_primary_key=bool(r[12]),
                        db_table=DbTableSimpleView(id=str(r[13]), name=r[14]),
                    )
                    results.append(
                        DbForeignKeySchemaView(
                            id=str(r[0]),
                            source_column=source_col,
                            target_column=target_col,
                        )
                    )
        logger.info(f"SchemaStore found {len(results)} foreign keys for database: '{database_id}'")
        return results

    def find_tables_by_ids(self, database_id: str, table_ids: list[str]) -> list[DbTableSchemaView]:
        """通过 Table ID 列表批量查询表结构及其所有列"""
        if not table_ids:
            return []
        sql = """
            SELECT
                t.id as table_id,
                t.name as table_name,
                t.description as table_desc,
                c.id as col_id,
                c.name as col_name,
                c.type as col_type,
                c.description as col_desc,
                c.is_primary_key as col_pk
            FROM db_table t
            LEFT JOIN db_column c ON t.id = c.table_id
            WHERE t.database_id = %(database_id)s AND t.id = ANY(%(table_ids)s::uuid[])
            ORDER BY t.name, c.name
        """
        return self._fetch_tables_with_columns(sql, {"database_id": database_id, "table_ids": table_ids})

    def find_tables_by_database_id_and_names(
        self, database_id: str, table_names: list[str]
    ) -> list[DbTableSchemaView]:
        """通过表名批量查询表结构及其所有列"""
        if not table_names:
            return []
        sql = """
            SELECT
                t.id as table_id,
                t.name as table_name,
                t.description as table_desc,
                c.id as col_id,
                c.name as col_name,
                c.type as col_type,
                c.description as col_desc,
                c.is_primary_key as col_pk
            FROM db_table t
            LEFT JOIN db_column c ON t.id = c.table_id
            WHERE t.database_id = %(database_id)s AND t.name = ANY(%(table_names)s)
            ORDER BY t.name, c.name
        """
        return self._fetch_tables_with_columns(sql, {"database_id": database_id, "table_names": table_names})

    def _fetch_tables_with_columns(self, sql: str, params: dict) -> list[DbTableSchemaView]:
        tables_map: dict[str, DbTableSchemaView] = {}
        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                for r in rows:
                    tid = str(r[0])
                    if tid not in tables_map:
                        tables_map[tid] = DbTableSchemaView(
                            id=tid,
                            name=r[1],
                            description=r[2],
                            columns=[],
                        )
                    if r[3] is not None:
                        col = DbColumnSchemaView(
                            id=str(r[3]),
                            name=r[4],
                            type=r[5],
                            description=r[6],
                            is_primary_key=bool(r[7]),
                            db_table=DbTableSimpleView(id=tid, name=r[1]),
                        )
                        tables_map[tid].columns.append(col)
        return list(tables_map.values())

