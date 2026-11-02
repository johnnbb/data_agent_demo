import pytest
from app.graph.nodes.evidence_recall import EvidenceRecallNode, EvidenceQueryRewrite
from app.graph.state import AgentState
from app.graph.spec import StateKey


@pytest.fixture
def recall_node():
    return EvidenceRecallNode()


def test_query_rewrite_denoise_and_context(recall_node):
    """
    测试 1：验证问题去噪与指代消解 (Query Rewrite)
    """
    multi_turn = "用户: 查一下加州阿拉米达县的学校\nAI: 已经为您查到 Alameda County 共有 300 所学校。"
    latest_query = "请问那里面符合免费午餐资格比例最高的是哪所学校呀？"

    rewritten = recall_node.rewrite_query(latest_query, multi_turn=multi_turn)
    print(f"\n[重写前] {latest_query}")
    print(f"[重写后] {rewritten}")

    assert isinstance(rewritten, str)
    assert len(rewritten) > 0
    assert "请问" not in rewritten
    assert any(keyword in rewritten for keyword in ["Alameda", "阿拉米达", "加州", "学校"])


def test_evidence_recall_node_full_pipeline(recall_node):
    """
    测试 2：验证 EvidenceRecallNode 完整节点的输入与状态产出 (对齐 StateKey.REWRITE_QUERY)
    """
    state: AgentState = {
        "input": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
        "databaseId": "california_schools",
        "MULTI_TURN_CONTEXT": "(无)",
    }

    output = recall_node(state)

    print("\n--- [重写后的查询 REWRITE_QUERY] ---")
    print(output[StateKey.REWRITE_QUERY.value])
    print("\n--- [装配好的业务依据 EVIDENCE] ---")
    print(output[StateKey.EVIDENCE.value])

    assert StateKey.REWRITE_QUERY.value in output
    assert len(output[StateKey.REWRITE_QUERY.value]) > 0

    assert StateKey.EVIDENCE.value in output
    evidence_text = output[StateKey.EVIDENCE.value]
    assert "### 业务术语表" in evidence_text
    assert "### 领域背景资料" in evidence_text

    assert "SELECT" in evidence_text
    assert "frpm" in evidence_text
