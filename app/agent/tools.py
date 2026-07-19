import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..rag.retriever import search_knowledge
from ..models.schemas import TicketDB, ConversationDB


async def create_support_ticket(session: AsyncSession, issue: str, title: str = "Support Ticket", customer_id: str | None = None) -> str:
    ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"

    ticket = TicketDB(
        id=ticket_id,
        title=title,
        description=issue,
        status="open",
        customer_id=customer_id,
    )
    session.add(ticket)
    await session.commit()

    return f"Support ticket created successfully. Ticket ID: {ticket_id}"


async def escalate_to_human(session: AsyncSession, reason: str, ticket_id: str | None = None) -> str:
    if ticket_id:
        result = await session.execute(select(TicketDB).where(TicketDB.id == ticket_id))
        ticket = result.scalar_one_or_none()
        if ticket:
            ticket.status = "escalated"
            await session.commit()

    return f"I have escalated this issue to a human support agent. Reason: {reason}. Someone will contact you shortly."


async def get_ticket(session: AsyncSession, ticket_id: str) -> str:
    result = await session.execute(select(TicketDB).where(TicketDB.id == ticket_id))
    ticket = result.scalar_one_or_none()

    if not ticket:
        return f"Ticket {ticket_id} not found."

    return (
        f"Ticket ID: {ticket.id}\n"
        f"Title: {ticket.title}\n"
        f"Issue: {ticket.description}\n"
        f"Status: {ticket.status}\n"
        f"Customer: {ticket.customer_id or 'N/A'}\n"
        f"Created: {ticket.created_at}\n"
        f"Updated: {ticket.updated_at}"
    )


def search_knowledge_tool(query: str) -> str:
    try:
        results = search_knowledge(query, k=4)

        if not results:
            return "I couldn't find specific information about that in our knowledge base."

        response = "Relevant information from our knowledge base:\n\n"
        for i, result in enumerate(results, 1):
            response += f"[{i}] {result['content']}\n\n"

        response += "I will use this to provide an accurate answer."
        return response
    except Exception as e:
        return f"Error accessing knowledge base: {str(e)}"
