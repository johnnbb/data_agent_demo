import pytest
from app.graph.spec import StateKey, NodeName
from app.graph.nodes.table_relation import (
    TableRelationNode,
    ColumnScore,
    table_relation_node,
)
from app.retrieval.schema_store import SchemaStore


def test_dynamic_threshold_selection_logic():
    """测试动态阈值平衡算法"""
    node = TableRelationNode()

    # 1. 达到 0.4 阈值的表 <= 4 张，直接返回 0.4
    table_scores = {
        "t1": 0.85,
        "t2": 0.65,
    }
    column_scores = [
        ColumnScore(table_id="t1", column_id="c1", score=0.8),
        ColumnScore(table_id="t2", column_id="c2", score=0.62),
    ]
    selection = node.select_threshold(table_scores, column_scores)
    assert selection.threshold == 0.4
    assert set(selection.table_ids) == {"t1", "t2"}

    # 2. 达到 0.4 阈值的表 > 4 张，动态提升阈值，逼近 4 张表
    table_scores_many = {
        f"t{i}": 0.40 + i * 0.05 for i in range(10) # 0.40 ~ 0.85
    }
    column_scores_many = [
        ColumnScore(table_id=f"t{i}", column_id=f"c{i}", score=0.40 + i * 0.05)
        for i in range(10)
    ]
    selection_many = node.select_threshold(table_scores_many, column_scores_many)
    assert selection_many.threshold > 0.4
    assert abs(4 - len(selection_many.table_ids)) <= 2

    # 3. 兜底逻辑：无任何表达到 0.4，取最高分 Top 1 表
    low_table_scores = {"t1": 0.1, "t2": 0.2}
    low_column_scores = [ColumnScore(table_id="t3", column_id="c1", score=0.3)]
    selection_low = node.select_threshold(low_table_scores, low_column_scores)
    assert "t2" in selection_low.table_ids or "t3" in selection_low.table_ids


def test_schema_store_relational_queries():
    """测试 SchemaStore 针对 PostgreSQL db_foreign_key 和 db_table 的查询能力"""
    store = SchemaStore()
    database_id = "california_schools"

    # 1. 查询外键
    foreign_keys = store.find_foreign_keys_by_database_id(database_id)
    assert len(foreign_keys) >= 2
    for fk in foreign_keys:
        assert fk.source_column.db_table is not None
        assert fk.target_column.db_table is not None
        assert "->" in f"{fk.source_column.db_table.name} -> {fk.target_column.db_table.name}"

    # 2. 按表名查询表结构及其列
    tables = store.find_tables_by_database_id_and_names(database_id, ["schools", "satscores"])
    assert len(tables) == 2
    table_names = {t.name for t in tables}
    assert "schools" in table_names
    assert "satscores" in table_names

    schools_tbl = next(t for t in tables if t.name == "schools")
    assert len(schools_tbl.columns) > 0
    # 验证主键识别
    pk_cols = [c.name for c in schools_tbl.columns if c.is_primary_key]
    assert "CDSCode" in pk_cols


def test_table_relation_node_e2e():
    """测试 TableRelationNode 节点端到端执行"""
    # 模拟上游节点产出的 State
    state = {
        "input": "What is the highest percentage of K-12 students eligible for free meals among schools in Alameda County?",
        "databaseId": "california_schools",
        StateKey.REWRITE_QUERY.value: "What is the highest percentage of K-12 students eligible for free meals among schools in Alameda County?",
        StateKey.EVIDENCE.value: "Alameda County refers to County = 'Alameda'",
        StateKey.TABLE_SCHEMA.value: [
            {
                "content": "Table: frpm",
                "score": 0.85,
                "metadata": {"tableId": "a60fd51f-feea-466c-baf1-2199ce4e7aa0"},
            }
        ],
        StateKey.COLUMN_SCHEMA.value: [
            {
                "content": "Column: County",
                "score": 0.90,
                "metadata": {
                    "tableId": "a22a3eda-9f0f-4605-bfe2-88da77e53904",
                    "columnId": "9d0263f3-0bf4-40fc-8086-4fdfd2b27bc9",
                },
            }
        ],
    }

    result = table_relation_node(state)
    assert StateKey.TABLE_RELATION.value in result

    relation_output = result[StateKey.TABLE_RELATION.value]
    assert relation_output["databaseId"] == "california_schools"
    assert "dbTables" in relation_output
    assert "dbForeignKeys" in relation_output

    table_names = [t["name"] for t in relation_output["dbTables"]]
    assert "schools" in table_names or "frpm" in table_names

