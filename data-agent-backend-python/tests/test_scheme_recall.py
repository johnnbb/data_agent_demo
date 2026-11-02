import pytest
from app.graph.nodes.scheme_recall import SchemeRecallNode
from app.graph.state import AgentState
from app.graph.spec import StateKey


@pytest.fixture
def scheme_node():
    return SchemeRecallNode()


def test_scheme_recall_node_tables_and_columns(scheme_node):
    """
    测试 SchemeRecallNode：验证能否从 california_schools 库召回相关的 Table 和 Column Schema
    """
    state: AgentState = {
        "REWRITE_QUERY": "查询加州阿拉米达县符合免费午餐资格比例最高的学校",
        "databaseId": "california_schools",
    }

    output = scheme_node(state)

    table_key = StateKey.TABLE_SCHEMA.value
    column_key = StateKey.COLUMN_SCHEMA.value

    print("\n--- [召回的候选表集合 (TABLE_SCHEME)] ---")
    for i, t in enumerate(output[table_key][:3], 1):
        print(f"Top {i} 表: {t['content'][:60]}... | 得分: {t['score']:.4f}")

    print("\n--- [召回的候选字段集合 (COLUMN_SCHEME)] ---")
    for i, c in enumerate(output[column_key][:5], 1):
        print(f"Top {i} 字段: {c['content']} | 得分: {c['score']:.4f}")

    # 1. 验证输出结构
    assert table_key in output
    assert column_key in output
    assert isinstance(output[table_key], list)
    assert isinstance(output[column_key], list)

    # 2. 验证是否召回了内容
    assert len(output[table_key]) > 0
    assert len(output[column_key]) > 0

    # 3. 验证相关性：表描述中应当包含关键的免费午餐表或学校表
    table_texts = [t["content"].lower() for t in output[table_key]]
    assert any("free and reduced-price meals" in text or "schools" in text for text in table_texts)
