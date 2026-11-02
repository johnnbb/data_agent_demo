import pytest
from app.retrieval import VectorRetriever, VectorType

@pytest.fixture
def retriever():
    return VectorRetriever()

def test_embed_query(retriever):
    vec = retriever.embed_query("加州学校免费午餐率")
    assert isinstance(vec, list)
    assert len(vec) == 1024

def test_retrieve_tables(retriever):
    query = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
    results = retriever.search_tables(query=query, database_id="california_schools", top_k=3)

    for i, r in enumerate(results, 1):
        print(f"\n[Top {i}] 相似度: {r.score:.4f} | 内容: {r.content} | 元数据: {r.metadata}")
    
    assert len(results) > 0
    # Top 结果应该包含免费午餐表或学校表
    table_names = [r.content.lower() for r in results]
    assert any("free and reduced-price meals" in name or "schools" in name for name in table_names)
    
    # 校验 metadata 规范
    top_meta = results[0].metadata
    assert top_meta.get("vectorType") == VectorType.TABLE.value
    assert top_meta.get("databaseId") == "california_schools"
    assert "tableId" in top_meta

def test_retrieve_few_shot_question(retriever):
    query = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
    results = retriever.search_questions(query=query, database_id="california_schools", top_k=3)
    
    assert len(results) > 0
    # 应当精准匹配到金标历史问题，相似度极高 (> 0.9)
    top_result = results[0]
    assert top_result.score > 0.9
    assert "Alameda County" in top_result.content
    assert top_result.metadata.get("vectorType") == VectorType.QUESTION_KNOWLEDGE.value

def test_retrieve_columns(retriever):
    query = "SchoolName and County"
    results = retriever.search_columns(query=query, database_id="california_schools", top_k=5)
    
    assert len(results) > 0
    for r in results:
        assert r.metadata.get("vectorType") == VectorType.COLUMN.value
        assert "columnId" in r.metadata
