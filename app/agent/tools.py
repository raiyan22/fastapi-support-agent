from typing import Dict
import uuid
from datetime import datetime

# Import RAG (correct relative import)
from ..rag.retriever import search_knowledge

# In-memory ticket storage
tickets: Dict[str, dict] = {}

def create_support_ticket(issue: str, customer_name: str = "Customer") -> str:
    """Create a support ticket and return the ticket ID."""
    ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"
    
    tickets[ticket_id] = {
        "id": ticket_id,
        "customer_name": customer_name,
        "issue": issue,
        "status": "open",
        "created_at": datetime.now().isoformat()
    }
    
    return f"Support ticket created successfully. Ticket ID: {ticket_id}"


def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent."""
    return f"I have escalated this issue to a human support agent. Reason: {reason}. Someone will contact you shortly."


def get_ticket(ticket_id: str) -> str:
    """Get information about a ticket."""
    ticket = tickets.get(ticket_id)
    if not ticket:
        return f"Ticket {ticket_id} not found."
    
    return (
        f"Ticket ID: {ticket['id']}\n"
        f"Customer: {ticket['customer_name']}\n"
        f"Issue: {ticket['issue']}\n"
        f"Status: {ticket['status']}\n"
        f"Created: {ticket['created_at']}"
    )


def search_knowledge_tool(query: str) -> str:
    """
    Search the company knowledge base for policies, FAQs, product info, etc.
    """
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