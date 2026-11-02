from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str]

class AgentInterface(BaseModel):
    transport: str
    url: str

class AgentCapabilities(BaseModel):
    streaming: bool = True
    pushNotifications: bool = True
    stateTransitionHistory: bool = True

class AgentCard(BaseModel):
    name: str = "sqlAgent"
    description: str = "专业的SQL生成Agent"
    url: str = "http://localhost:3500/api/a2a/jsonrpc"
    version: str = "1.0"
    capabilities: AgentCapabilities = AgentCapabilities()
    defaultInputModes: List[str] = ["text/plain"]
    defaultOutputModes: List[str] = ["text/plain"]
    skills: List[AgentSkill] = [
        AgentSkill(id="sql", name="sql generator", description="Generate a SQL query", tags=["sql", "query"])
    ]
    supportsAuthenticatedExtendedCard: bool = False
    additionalInterfaces: List[AgentInterface] = [
        AgentInterface(transport="JSONRPC", url="http://localhost:3500/api/a2a/jsonrpc")
    ]
    preferredTransport: str = "JSONRPC"
    protocolVersion: str = "0.3.0"

class MessagePart(BaseModel):
    kind: str
    text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class Message(BaseModel):
    messageId: Optional[str] = None
    role: Optional[str] = "user"
    kind: Optional[str] = "message"
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    parts: List[MessagePart] = []

class MessageSendParams(BaseModel):
    message: Message

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Any] = None
    method: str
    params: Optional[Dict[str, Any]] = None
