from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from typing import Dict, List

from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse

app = FastAPI(
    title="Local AI Customer Support Agent",
    description="Fully local AI Customer Support Agent using llama.cpp + Phi-4-mini",
    version="0.2.0"
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

# Simple in-memory conversation storage
conversations: Dict[str, List[dict]] = {}

SYSTEM_PROMPT = """You are a professional and helpful Customer Support Agent.
Your name is Alex.
Be polite, clear, and concise.
If you don't know something, say so honestly and offer to escalate.
Always try to solve the customer's problem."""

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
        # Get or create conversation history
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

        # Call the local model
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=conversations[conv_id],
            temperature=0.6,
            max_tokens=600,
        )

        reply = response.choices[0].message.content

        # Save assistant reply
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