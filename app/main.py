from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from typing import Dict, List
from contextlib import asynccontextmanager
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse, ConversationDB
from app.db import init_db, get_db
from app.routers.documents import router as documents_router

from app.agent.tools import (
    create_support_ticket,
    escalate_to_human,
    get_ticket,
    search_knowledge_tool,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Local AI Customer Support Agent",
    description="Fully local AI Customer Support Agent using llama.cpp + Phi-4-mini + RAG",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)

client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
)

conversations: Dict[str, List[dict]] = {}

SYSTEM_PROMPT = """You are Alex, a customer support agent. You MUST use tools — never just talk about using them.

WHEN THE CUSTOMER HAS A PROBLEM (complaint, bug, broken item, issue):
You MUST output EXACTLY this on its own line (nothing else):
TOOL: create_support_ticket | issue: <short description>

WHEN THE CUSTOMER ASKS ABOUT POLICIES, SHIPPING, RETURNS, FAQS, OR ANY COMPANY INFO:
You MUST output EXACTLY this on its own line (nothing else):
TOOL: search_knowledge_tool | query: <search keywords>

WHEN THE CUSTOMER ASKS FOR A HUMAN OR THE ISSUE IS TOO COMPLEX:
You MUST output EXACTLY this on its own line (nothing else):
TOOL: escalate_to_human | reason: <why>

WHEN THE CUSTOMER ASKS ABOUT A TICKET OR PROVIDES A TICKET ID:
You MUST output EXACTLY this on its own line (nothing else):
TOOL: get_ticket | ticket_id: <id>

EXAMPLES OF CORRECT BEHAVIOR:

Customer: My order arrived broken
Assistant: TOOL: create_support_ticket | issue: Order arrived broken

Customer: What is your return policy?
Assistant: TOOL: search_knowledge_tool | query: return policy

Customer: I want to talk to a real person
Assistant: TOOL: escalate_to_human | reason: Customer requested human agent

Customer: What is the status of TICKET-ABC123?
Assistant: TOOL: get_ticket | ticket_id: TICKET-ABC123

RULES:
- Output ONLY the TOOL line when using a tool. Nothing before, nothing after.
- Do not say "I will create a ticket" — just output the TOOL line.
- Do not ask for more details — create the ticket with whatever info the customer gave you.
- After receiving a tool result, give a short 1-2 sentence friendly reply.
"""


def parse_tool_call(text: str):
    text = text.strip()

    match = re.search(r"TOOL:\s*search_knowledge_tool\s*\|\s*query:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "search_knowledge_tool", {"query": match.group(1).strip()}

    match = re.search(r"TOOL:\s*create_support_ticket\s*\|\s*issue:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "create_support_ticket", {"issue": match.group(1).strip()}

    match = re.search(r"TOOL:\s*escalate_to_human\s*\|\s*reason:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "escalate_to_human", {"reason": match.group(1).strip()}

    match = re.search(r"TOOL:\s*get_ticket\s*\|\s*ticket_id:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "get_ticket", {"ticket_id": match.group(1).strip()}

    return None, None


async def execute_tool(session: AsyncSession, tool_name: str, params: dict) -> str:
    if tool_name == "search_knowledge_tool":
        return search_knowledge_tool(query=params["query"])
    elif tool_name == "create_support_ticket":
        return await create_support_ticket(session, issue=params["issue"])
    elif tool_name == "escalate_to_human":
        return await escalate_to_human(session, reason=params["reason"])
    elif tool_name == "get_ticket":
        return await get_ticket(session, ticket_id=params["ticket_id"])
    return "Unknown tool"


async def save_conversation(session: AsyncSession, session_id: str, user_message: str, assistant_message: str, ticket_id: str | None = None):
    entry = ConversationDB(
        session_id=session_id,
        user_message=user_message,
        assistant_message=assistant_message,
        ticket_id=ticket_id,
    )
    session.add(entry)
    await session.commit()


@app.get("/")
def root():
    return {
        "message": "Local AI Customer Support Agent with RAG is running",
        "model": settings.LLM_MODEL,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session: AsyncSession = Depends(get_db)):
    try:
        conv_id = request.conversation_id or "default"

        if conv_id not in conversations:
            conversations[conv_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        conversations[conv_id].append({
            "role": "user",
            "content": request.message,
        })

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=conversations[conv_id],
            temperature=0.4,
            max_tokens=600,
        )

        reply = response.choices[0].message.content.strip()
        tool_name, params = parse_tool_call(reply)

        if tool_name:
            tool_result = await execute_tool(session, tool_name, params)

            conversations[conv_id].append({"role": "assistant", "content": reply})
            conversations[conv_id].append({"role": "user", "content": f"Tool result: {tool_result}"})

            final_response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=conversations[conv_id],
                temperature=0.5,
                max_tokens=600,
            )

            final_reply = final_response.choices[0].message.content.strip()
            conversations[conv_id].append({"role": "assistant", "content": final_reply})

            await save_conversation(session, conv_id, request.message, final_reply)
            return ChatResponse(reply=final_reply, conversation_id=conv_id)

        conversations[conv_id].append({"role": "assistant", "content": reply})
        await save_conversation(session, conv_id, request.message, reply)
        return ChatResponse(reply=reply, conversation_id=conv_id)

    except Exception as e:
        return ChatResponse(
            reply=f"Sorry, something went wrong: {str(e)}",
            conversation_id=request.conversation_id,
        )


@app.get("/chat/{session_id}")
async def get_conversation(session_id: str, session: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await session.execute(
        select(ConversationDB).where(ConversationDB.session_id == session_id).order_by(ConversationDB.timestamp)
    )
    rows = result.scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No messages found for this session")
    return [
        {"user": r.user_message, "assistant": r.assistant_message, "timestamp": str(r.timestamp)}
        for r in rows
    ]


@app.delete("/chat/{session_id}")
async def delete_conversation(session_id: str, session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        delete(ConversationDB).where(ConversationDB.session_id == session_id)
    )
    await session.commit()
    in_memory = conversations.pop(session_id, None)
    return {"deleted_db_rows": result.rowcount, "deleted_in_memory": in_memory is not None}
