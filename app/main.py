from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from typing import Dict, List
from contextlib import asynccontextmanager
from pathlib import Path
import re
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, select

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — creating DB tables if needed")
    await init_db()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Local AI Customer Support Agent",
    description="Fully local AI Customer Support Agent using llama.cpp + Phi-4-mini + RAG",
    version="0.6.0",
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

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
    try:
        if tool_name == "search_knowledge_tool":
            return search_knowledge_tool(query=params["query"])
        elif tool_name == "create_support_ticket":
            return await create_support_ticket(session, issue=params["issue"])
        elif tool_name == "escalate_to_human":
            return await escalate_to_human(session, reason=params["reason"])
        elif tool_name == "get_ticket":
            return await get_ticket(session, ticket_id=params["ticket_id"])
        return "Unknown tool"
    except Exception as e:
        logger.error(f"Tool execution failed ({tool_name}): {e}")
        return f"Error executing {tool_name}: {str(e)}"


async def save_conversation(session: AsyncSession, session_id: str, user_message: str, assistant_message: str, customer_id: str | None = None, ticket_id: str | None = None):
    entry = ConversationDB(
        session_id=session_id,
        customer_id=customer_id,
        user_message=user_message,
        assistant_message=assistant_message,
        ticket_id=ticket_id,
    )
    session.add(entry)
    await session.commit()
    logger.info(f"Saved conversation: session={session_id} customer={customer_id}")


async def load_customer_history(session: AsyncSession, customer_id: str, limit: int = 5) -> str:
    result = await session.execute(
        select(ConversationDB)
        .where(ConversationDB.customer_id == customer_id)
        .order_by(ConversationDB.timestamp.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    if not rows:
        return ""

    lines = [f"Previous conversation with this customer:"]
    for r in reversed(rows):
        lines.append(f"Customer: {r.user_message}")
        lines.append(f"Alex: {r.assistant_message}")
    return "\n".join(lines)


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session: AsyncSession = Depends(get_db)):
    try:
        conv_id = request.conversation_id or "default"
        customer_id = request.customer_id

        if conv_id not in conversations:
            history_block = ""
            if customer_id:
                history_block = await load_customer_history(session, customer_id)
                if history_block:
                    logger.info(f"Loaded history for customer {customer_id}")

            system_content = SYSTEM_PROMPT
            if history_block:
                system_content += f"\n\n{history_block}"

            conversations[conv_id] = [
                {"role": "system", "content": system_content}
            ]

        conversations[conv_id].append({
            "role": "user",
            "content": request.message,
        })

        logger.info(f"Chat request: session={conv_id} customer={customer_id} message={request.message[:80]}")

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=conversations[conv_id],
            temperature=0.4,
            max_tokens=600,
        )

        reply = response.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply[:120]}")
        tool_name, params = parse_tool_call(reply)

        if tool_name:
            logger.info(f"Tool call: {tool_name} params={params}")
            tool_result = await execute_tool(session, tool_name, params)
            logger.info(f"Tool result: {tool_result[:120]}")

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

            await save_conversation(session, conv_id, request.message, final_reply, customer_id)
            return ChatResponse(reply=final_reply, conversation_id=conv_id)

        conversations[conv_id].append({"role": "assistant", "content": reply})
        await save_conversation(session, conv_id, request.message, reply, customer_id)
        return ChatResponse(reply=reply, conversation_id=conv_id)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(
            reply=f"Sorry, something went wrong: {str(e)}",
            conversation_id=request.conversation_id,
        )


@app.get("/chat/{session_id}")
async def get_conversation(session_id: str, session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        select(ConversationDB).where(ConversationDB.session_id == session_id).order_by(ConversationDB.timestamp)
    )
    rows = result.scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No messages found for this session")
    return [
        {"user": r.user_message, "assistant": r.assistant_message, "customer_id": r.customer_id, "timestamp": str(r.timestamp)}
        for r in rows
    ]


@app.delete("/chat/{session_id}")
async def delete_conversation(session_id: str, session: AsyncSession = Depends(get_db)):
    result = await session.execute(
        delete(ConversationDB).where(ConversationDB.session_id == session_id)
    )
    await session.commit()
    in_memory = conversations.pop(session_id, None)
    logger.info(f"Deleted conversation: session={session_id} db_rows={result.rowcount}")
    return {"deleted_db_rows": result.rowcount, "deleted_in_memory": in_memory is not None}
