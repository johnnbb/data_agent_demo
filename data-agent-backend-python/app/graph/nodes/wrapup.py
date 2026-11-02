from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.nodes.travel import get_llm
from app.graph.state import AgentState

async def wrap_up_node(state: AgentState) -> dict:
    llm = get_llm()
    scene_label = state.get("sceneLabel", "内容规划")
    draft = state.get("draft", "")
    messages = [
        SystemMessage(content="你是一个教程里的收尾节点。请把输入整理成更通俗的行动清单，控制在 3 到 5 条，每条一句话。"),
        HumanMessage(content=f"请把这份{scene_label}整理成用户一看就能执行的清单：{draft}")
    ]
    response = await llm.ainvoke(messages)
    return {"finalOutput": str(response.content)}
