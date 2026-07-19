from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from typing import Dict, List
import re

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse
from app.agent.tools import create_support_ticket, escalate_to_human, get_ticket

app = FastAPI(
    title="Local AI Customer Support Agent",
    description="Fully local AI Customer Support Agent using llama.cpp + Phi-4-mini",
    version="0.3.0"
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

You can use the following tools when needed:

1. create_support_ticket
   - Use when the customer has a problem that needs to be tracked.
   - Format: TOOL: create_support_ticket | issue: <description>

2. escalate_to_human
   - Use when you cannot solve the issue or the customer asks for a human.
   - Format: TOOL: escalate_to_human | reason: <reason>

3. get_ticket
   - Use when the customer asks about a ticket status.
   - Format: TOOL: get_ticket | ticket_id: <id>

Rules:
- Be polite, clear and professional.
- Only use a tool when necessary.
- If you use a tool, output ONLY the tool call in the format above (nothing else).
- After receiving the tool result, give a helpful final answer to the customer.
"""

def parse_tool_call(text: str):
    """Parse tool call from model response"""
    text = text.strip()

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
    if tool_name == "create_support_ticket":
        return create_support_ticket(issue=params["issue"])
    elif tool_name == "escalate_to_human":
        return escalate_to_human(reason=params["reason"])
    elif tool_name == "get_ticket":
        return get_ticket(ticket_id=params["ticket_id"])
    return "Unknown tool"

@app.get("/")
def root():
    return {
        "message": "Local AI Customer Support Agent is running",
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

        # Add user message
        conversations[conv_id].append({
            "role": "user",
            "content": request.message
        })

        # First call to the model
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=conversations[conv_id],
            temperature=0.4,
            max_tokens=500,
        )

        reply = response.choices[0].message.content.strip()

        # Check if the model wants to use a tool
        tool_name, params = parse_tool_call(reply)

        if tool_name:
            # Execute the tool
            tool_result = execute_tool(tool_name, params)

            # Add tool result to conversation
            conversations[conv_id].append({
                "role": "assistant",
                "content": reply
            })
            conversations[conv_id].append({
                "role": "user",
                "content": f"Tool result: {tool_result}"
            })

            # Second call so the model can give a final answer
            final_response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=conversations[conv_id],
                temperature=0.5,
                max_tokens=500,
            )

            final_reply = final_response.choices[0].message.content.strip()

            conversations[conv_id].append({
                "role": "assistant",
                "content": final_reply
            })

            return ChatResponse(
                reply=final_reply,
                conversation_id=conv_id
            )

        # No tool was used
        conversations[conv_id].append({
            "role": "assistant",
            "content": reply
        })

        return ChatResponse(
            reply=reply,
            conversation_id=conv_id
        )

    except Exception as e:
        return ChatResponse(
            reply=f"Sorry, something went wrong: {str(e)}",
            conversation_id=request.conversation_id
        )