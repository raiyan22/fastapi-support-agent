from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from typing import Dict, List
import re

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse

# Import all tools
from app.agent.tools import (
    create_support_ticket, 
    escalate_to_human, 
    get_ticket,
    search_knowledge_tool
)

app = FastAPI(
    title="Local AI Customer Support Agent",
    description="Fully local AI Customer Support Agent using llama.cpp + Phi-4-mini + RAG",
    version="0.4.0"
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

# In-memory conversation storage
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
    """Parse tool call from model response"""
    text = text.strip()

    # Search knowledge
    match = re.search(r"TOOL:\s*search_knowledge_tool\s*\|\s*query:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "search_knowledge_tool", {"query": match.group(1).strip()}

    # create_support_ticket
    match = re.search(r"TOOL:\s*create_support_ticket\s*\|\s*issue:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "create_support_ticket", {"issue": match.group(1).strip()}

    # escalate_to_human
    match = re.search(r"TOOL:\s*escalate_to_human\s*\|\s*reason:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "escalate_to_human", {"reason": match.group(1).strip()}

    # get_ticket
    match = re.search(r"TOOL:\s*get_ticket\s*\|\s*ticket_id:\s*(.+)", text, re.IGNORECASE)
    if match:
        return "get_ticket", {"ticket_id": match.group(1).strip()}

    return None, None


def execute_tool(tool_name: str, params: dict) -> str:
    if tool_name == "search_knowledge_tool":
        return search_knowledge_tool(query=params["query"])
    elif tool_name == "create_support_ticket":
        return create_support_ticket(issue=params["issue"])
    elif tool_name == "escalate_to_human":
        return escalate_to_human(reason=params["reason"])
    elif tool_name == "get_ticket":
        return get_ticket(ticket_id=params["ticket_id"])
    return "Unknown tool"


@app.get("/")
def root():
    return {
        "message": "Local AI Customer Support Agent with RAG is running",
        "model": settings.LLM_MODEL,
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        conv_id = request.conversation_id or "default"

        if conv_id not in conversations:
            conversations[conv_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

        conversations[conv_id].append({
            "role": "user",
            "content": request.message
        })

        # Call LLM
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=conversations[conv_id],
            temperature=0.4,
            max_tokens=600,
        )

        reply = response.choices[0].message.content.strip()

        tool_name, params = parse_tool_call(reply)

        if tool_name:
            tool_result = execute_tool(tool_name, params)

            # Add to history
            conversations[conv_id].append({"role": "assistant", "content": reply})
            conversations[conv_id].append({"role": "user", "content": f"Tool result: {tool_result}"})

            # Final response
            final_response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=conversations[conv_id],
                temperature=0.5,
                max_tokens=600,
            )

            final_reply = final_response.choices[0].message.content.strip()
            conversations[conv_id].append({"role": "assistant", "content": final_reply})

            return ChatResponse(reply=final_reply, conversation_id=conv_id)

        # No tool used
        conversations[conv_id].append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply, conversation_id=conv_id)

    except Exception as e:
        return ChatResponse(
            reply=f"Sorry, something went wrong: {str(e)}",
            conversation_id=request.conversation_id
        )