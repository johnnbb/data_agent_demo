import pytest
from langgraph.graph import END

from app.graph.spec import NodeName, StateKey
from app.graph.nodes.sql_generator import SqlGeneratorNode, extract_raw_sql
from app.graph.nodes.sql_execute import SqlExecuteNode, locate_sqlite_database
from app.graph.edges import plan_executor_edge
from app.graph.builder import build_graph


def test_extract_raw_sql():
    raw_markdown = """```sql
    SELECT * FROM frpm WHERE County = 'Alameda';
    ```"""
    cleaned = extract_raw_sql(raw_markdown)
    assert cleaned.startswith("SELECT")
    assert cleaned.endswith(";")
    assert "```" not in cleaned

    plain_sql = "SELECT id, name FROM schools"
    cleaned_plain = extract_raw_sql(plain_sql)
    assert cleaned_plain == "SELECT id, name FROM schools;"


def test_locate_sqlite_database():
    db_path = locate_sqlite_database("california_schools")
    assert db_path is not None
    assert db_path.exists()
    assert db_path.name == "california_schools.sqlite"


def test_sql_execute_node_real_query():
    node = SqlExecuteNode()
    query = 'SELECT "CDSCode", "County Name" FROM frpm WHERE "County Name" = \'Alameda\' LIMIT 5;'
    state = {
        StateKey.SQL_GENERATION_RESULT.value: query,
        StateKey.DATABASE_ID.value: "california_schools",
        StateKey.CURRENT_STEP.value: 1,
    }

    result = node(state)

    # 1. 验证步数安全推进
    assert result[StateKey.CURRENT_STEP.value] == 2

    # 2. 验证执行输出
    exec_result = result[StateKey.SQL_EXECUTION_RESULT.value]
    result_set = exec_result["resultSet"]
    display = exec_result["display"]

    assert result_set["errorMsg"] is None
    assert "CDSCode" in result_set["column"]
    assert "County Name" in result_set["column"]
    assert len(result_set["data"]) == 5
    assert result_set["data"][0]["County Name"] == "Alameda"

    # 3. 验证可视化规格
    assert display["type"] == "table"
    assert display["x"] == "CDSCode"
    assert "County Name" in display["y"]


def test_sql_execute_multi_step_accumulation():
    """
    专门针对【执行输出覆盖漏洞】的防御性测试：
    模拟连续执行第 1 步与第 2 步，断言 step_1 与 step_2 能够被安全累积合并，绝不相互覆盖。
    """
    node = SqlExecuteNode()

    # 第 1 步：查餐食
    state_step1 = {
        StateKey.SQL_GENERATION_RESULT.value: 'SELECT "County Name" FROM frpm LIMIT 2;',
        StateKey.DATABASE_ID.value: "california_schools",
        StateKey.CURRENT_STEP.value: 1,
    }
    result_step1 = node(state_step1)
    assert result_step1[StateKey.CURRENT_STEP.value] == 2
    assert "step_1" in result_step1[StateKey.EXECUTION_OUTPUT.value]

    # 第 2 步：拿着第 1 步的状态产物，查学校总数
    state_step2 = {
        StateKey.SQL_GENERATION_RESULT.value: "SELECT COUNT(*) AS total FROM schools;",
        StateKey.DATABASE_ID.value: "california_schools",
        StateKey.CURRENT_STEP.value: result_step1[StateKey.CURRENT_STEP.value],
        StateKey.EXECUTION_OUTPUT.value: result_step1[StateKey.EXECUTION_OUTPUT.value],
    }
    result_step2 = node(state_step2)
    assert result_step2[StateKey.CURRENT_STEP.value] == 3

    merged_output = result_step2[StateKey.EXECUTION_OUTPUT.value]
    # 核心检验：两步数据必须同时存在于状态字典中，互不覆盖！
    assert "step_1" in merged_output
    assert "step_2" in merged_output
    assert len(merged_output["step_1"]["data"]) == 2
    assert "total" in merged_output["step_2"]["data"][0]



def test_sql_execute_node_error_handling():
    node = SqlExecuteNode()
    bad_query = "SELECT non_existent_col FROM definitely_not_a_table;"
    state = {
        StateKey.SQL_GENERATION_RESULT.value: bad_query,
        StateKey.DATABASE_ID.value: "california_schools",
        StateKey.CURRENT_STEP.value: 1,
    }

    result = node(state)
    assert result[StateKey.CURRENT_STEP.value] == 2

    exec_result = result[StateKey.SQL_EXECUTION_RESULT.value]
    result_set = exec_result["resultSet"]
    assert result_set["errorMsg"] is not None
    assert "no such table" in result_set["errorMsg"]
    assert len(result_set["data"]) == 0


def test_plan_executor_edge():
    assert plan_executor_edge({StateKey.NEXT_NODE.value: NodeName.SQL_GENERATION.value}) == NodeName.SQL_GENERATION.value
    assert plan_executor_edge({StateKey.NEXT_NODE.value: END}) == END
    assert plan_executor_edge({}) == END


@pytest.mark.asyncio
async def test_full_pipeline_with_sql_generate_and_execute():
    """
    全流程端到端测试：
    1. 用户提问触发检索、可行性、规划
    2. 停留在 HUMAN_FEEDBACK 节点挂起
    3. 人工批准继续执行
    4. 执行流进入 PLAN_EXECUTION -> SQL_GENERATION -> SQL_EXECUTE -> END
    5. 验证最终产物包含生成的真实 SQL 与从 SQLite 查询到的真实数据集
    """
    graph = build_graph(interrupt_before_feedback=True)
    thread_id = "test-sql-gen-exec-pipeline-1"
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "input": "What is the highest percentage of K–12 students eligible for free meals among schools in Alameda County?",
        "databaseId": "california_schools",
    }

    # 1. 运行第一阶段（停留在 HUMAN_FEEDBACK 之前）
    first_nodes = []
    async for mode, event in graph.astream(initial_input, config, stream_mode=["updates"]):
        if mode == "updates":
            first_nodes.extend(list(event.keys()))

    assert NodeName.PLANNER.value in first_nodes
    assert NodeName.SQL_GENERATION.value not in first_nodes

    snapshot = await graph.aget_state(config)
    assert snapshot.next == (NodeName.HUMAN_FEEDBACK.value,)

    # 2. 模拟人工审核批准
    await graph.aupdate_state(
        config,
        {
            StateKey.CONFIRMATION_APPROVED.value: True,
            StateKey.CONFIRMATION_FEEDBACK.value: "确认通过",
        },
    )

    # 3. 恢复运行第二阶段
    second_nodes = []
    async for mode, event in graph.astream(None, config, stream_mode=["updates"]):
        if mode == "updates":
            second_nodes.extend(list(event.keys()))

    assert NodeName.HUMAN_FEEDBACK.value in second_nodes
    assert NodeName.PLAN_EXECUTION.value in second_nodes
    assert NodeName.SQL_GENERATION.value in second_nodes
    assert NodeName.SQL_EXECUTION.value in second_nodes

    # 4. 验证最终状态数据
    final_snapshot = await graph.aget_state(config)
    values = final_snapshot.values

    # SQL 生成产物
    sql = values.get(StateKey.SQL_GENERATION_RESULT.value)
    assert sql is not None
    assert "SELECT" in sql.upper()
    assert "FRPM" in sql.upper()

    # SQL 执行产物
    exec_result = values.get(StateKey.SQL_EXECUTION_RESULT.value)
    assert exec_result is not None
    assert "resultSet" in exec_result
    result_set = exec_result["resultSet"]
    assert result_set["errorMsg"] is None
    assert len(result_set["column"]) > 0
    assert len(result_set["data"]) > 0

