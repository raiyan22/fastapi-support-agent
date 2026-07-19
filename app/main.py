from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from typing import Dict, List
from contextlib import asynccontextmanager
import re
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse
from app.db import init_db, get_db

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

client = OpenAI(
    base_url=settings.LLM_BASE_URL,
    api_key=settings.LLM_API_KEY,
)

conversations: Dict[str, List[dict]] = {}

SYSTEM_PROMPT = """You are a professional Customer Support Agent named Alex.

You have access to these tools:

1. search_knowledge_tool
   - Use when the customer asks about policies, returns, shipping, products, FAQs, etc.
   - Format: TOOL: search_knowledge_tool | query: <question>

2. create_support_ticket
   - Use when the customer has a problem that needs tracking.
   - Format: TOOL: create_support_ticket | issue: <description>

3. escalate_to_human
   - Use when issue is complex or customer requests human.
   - Format: TOOL: escalate_to_human | reason: <reason>

4. get_ticket
   - Use when customer asks about existing ticket status.
   - Format: TOOL: get_ticket | ticket_id: <id>

Rules:
- Be polite, helpful, and professional.
- First try to answer using search_knowledge_tool if it's about company info.
- Only output the TOOL call when you decide to use a tool. Do not add extra text.
- After getting tool result, give a friendly final response.
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

            return ChatResponse(reply=final_reply, conversation_id=conv_id)

        conversations[conv_id].append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply, conversation_id=conv_id)

    except Exception as e:
        return ChatResponse(
            reply=f"Sorry, something went wrong: {str(e)}",
            conversation_id=request.conversation_id,
        )
