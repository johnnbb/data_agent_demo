import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.memory import MemorySaver

from app.a2a.models import AgentCard
from app.graph.builder import build_graph
from app.graph.spec import NodeName, StateKey

router = APIRouter()

# 全局共享的检查点存储器与图实例
memory_saver = MemorySaver()
graph = build_graph(checkpointer=memory_saver, interrupt_before_feedback=True)

# 内存任务存储 (TaskStore)
tasks: Dict[str, Dict[str, Any]] = {}


@router.get("/.well-known/agent-card.json")
async def get_agent_card():
    return AgentCard().model_dump()


@router.post("/a2a/jsonrpc")
async def handle_jsonrpc(request: Request):
    body = await request.json()
    req_id = body.get("id")
    method = body.get("method")

    if method != "message/stream":
        return JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"},
            },
        )

    params = body.get("params", {})
    msg = params.get("message", {})
    task_id = msg.get("taskId")
    context_id = msg.get("contextId") or str(uuid.uuid4())
    metadata = msg.get("metadata") or {}

    async def sse_event_stream() -> AsyncGenerator[str, None]:
        def make_sse(result_data: dict) -> str:
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result_data,
            }
            return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"

        existing_task = tasks.get(task_id) if task_id else None

        # =============================================================
        # 分支 1：恢复现有中断任务（Human-in-the-Loop Resume）
        # =============================================================
        if existing_task is not None:
            current_task_id = existing_task["id"]
            current_context_id = existing_task.get("contextId", context_id)
            artifact_counter = existing_task.get("artifact_counter", 0)

            def next_resume_artifact_id() -> str:
                nonlocal artifact_counter
                artifact_counter += 1
                return str(artifact_counter)

            config = {"configurable": {"thread_id": current_task_id}}
            approved = metadata.get(StateKey.CONFIRMATION_APPROVED.value, True)
            feedback = metadata.get(StateKey.CONFIRMATION_FEEDBACK.value, "")

            # 将人工审核输入写入状态
            graph.update_state(
                config,
                {
                    StateKey.CONFIRMATION_APPROVED.value: approved,
                    StateKey.CONFIRMATION_FEEDBACK.value: feedback,
                },
            )

            # 以 None 输入恢复图的执行
            async for mode, event in graph.astream(
                None, config, stream_mode=["updates"]
            ):
                if mode == "updates":
                    update_dict = event
                    for node_name, node_data in update_dict.items():
                        parts = [{"data": node_data, "kind": "data"}]
                        if isinstance(node_data, dict):
                            report_text = node_data.get(StateKey.REPORT_RESULT.value)
                            if report_text:
                                parts.append({"text": str(report_text), "kind": "text"})

                        yield make_sse(
                            {
                                "taskId": current_task_id,
                                "contextId": current_context_id,
                                "kind": "artifact-update",
                                "artifact": {
                                    "artifactId": next_resume_artifact_id(),
                                    "name": node_name,
                                    "parts": parts,
                                    "metadata": {"outputType": "GRAPH_NODE_FINISHED"},
                                },
                            }
                        )

            existing_task["artifact_counter"] = artifact_counter

            # 检查是否再次中断（例如驳回后重新规划再次到达 HUMAN_FEEDBACK）
            state_snapshot = graph.get_state(config)
            if (
                state_snapshot.next
                and NodeName.HUMAN_FEEDBACK.value in state_snapshot.next
            ):
                yield make_sse(
                    {
                        "taskId": current_task_id,
                        "contextId": current_context_id,
                        "kind": "status-update",
                        "status": {
                            "state": "input-required",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    }
                )
            else:
                yield make_sse(
                    {
                        "taskId": current_task_id,
                        "contextId": current_context_id,
                        "final": True,
                        "kind": "status-update",
                        "status": {
                            "state": "completed",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    }
                )
            return

        # =============================================================
        # 分支 2：首次执行全新任务（New Task Submission）
        # =============================================================
        current_task_id = task_id or str(uuid.uuid4())
        tasks[current_task_id] = {
            "id": current_task_id,
            "contextId": context_id,
            "artifact_counter": 0,
        }
        config = {"configurable": {"thread_id": current_task_id}}

        artifact_counter = 0

        def next_artifact_id() -> str:
            nonlocal artifact_counter
            artifact_counter += 1
            return str(artifact_counter)

        input_text = ""
        for part in msg.get("parts", []):
            if part.get("kind") == "text":
                input_text = part.get("text", "")

        if not input_text:
            if isinstance(msg.get("content"), str):
                input_text = msg.get("content", "")
            elif isinstance(msg.get("text"), str):
                input_text = msg.get("text", "")

        database_id = metadata.get("databaseId", "california_schools")

        # 1. 提交初始任务事件
        yield make_sse(
            {
                "id": current_task_id,
                "contextId": context_id,
                "kind": "task",
                "status": {
                    "state": "submitted",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "artifacts": [],
                "history": [msg],
            }
        )

        initial_state = {
            "input": input_text,
            "databaseId": database_id,
        }

        # 2. 流式调度执行图节点
        async for mode, event in graph.astream(
            initial_state, config, stream_mode=["updates"]
        ):
            if mode == "updates":
                update_dict = event
                for node_name, node_data in update_dict.items():
                    parts = [{"data": node_data, "kind": "data"}]
                    if isinstance(node_data, dict):
                        report_text = node_data.get(StateKey.REPORT_RESULT.value)
                        if report_text:
                            parts.append({"text": str(report_text), "kind": "text"})

                    yield make_sse(
                        {
                            "taskId": current_task_id,
                            "contextId": context_id,
                            "kind": "artifact-update",
                            "artifact": {
                                "artifactId": next_artifact_id(),
                                "name": node_name,
                                "parts": parts,
                                "metadata": {"outputType": "GRAPH_NODE_FINISHED"},
                            },
                        }
                    )

        tasks[current_task_id]["artifact_counter"] = artifact_counter

        # 3. 检查当前图的状态是否因 interruptBefore 暂停
        state_snapshot = graph.get_state(config)
        if (
            state_snapshot.next
            and NodeName.HUMAN_FEEDBACK.value in state_snapshot.next
        ):
            yield make_sse(
                {
                    "taskId": current_task_id,
                    "contextId": context_id,
                    "kind": "status-update",
                    "status": {
                        "state": "input-required",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
        else:
            yield make_sse(
                {
                    "taskId": current_task_id,
                    "contextId": context_id,
                    "final": True,
                    "kind": "status-update",
                    "status": {
                        "state": "completed",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")
