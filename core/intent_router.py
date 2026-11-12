
def detect_intent(user_input):
    """Identify the user's intention from their request."""

    text = user_input.lower()

    # Viva-related requests
    if any(word in text for word in [
        "viva", "question", "quiz", "interview"
    ]):
        return "viva"

    # Concept extraction requests
    elif any(word in text for word in [
        "concept", "algorithm", "methodology",
        "technique", "technology", "findings",
        "key points"
    ]):
        return "concept"

    # Summary-related requests
    elif any(word in text for word in [
        "summarize", "summary", "overview",
        "research problem", "objectives",
        "conclusion"
    ]):
        return "summary"

    # General paper questions
    else:
        return "chat"


def route_intent(user_input):
    """Return the appropriate agent name."""

    intent = detect_intent(user_input)

    routes = {
        "summary": "Summarizer Agent",
        "concept": "Concept Extraction Agent",
        "viva": "Viva Agent",
        "chat": "RAG Question Answering"
    }

    return routes[intent]
