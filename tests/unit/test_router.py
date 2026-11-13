
from core.intent_router import detect_intent, route_intent


# Test summary intent
def test_summary_intent():
    assert detect_intent("Summarize this research paper") == "summary"
    assert detect_intent("Give me an overview") == "summary"


# Test concept extraction intent
def test_concept_intent():
    assert detect_intent("Explain the key concepts") == "concept"
    assert detect_intent("What algorithms are used?") == "concept"


# Test viva intent
def test_viva_intent():
    assert detect_intent("Generate viva questions") == "viva"
    assert detect_intent("Prepare me for an interview") == "viva"


# Test general chat intent
def test_chat_intent():
    assert detect_intent("What is the main contribution of this paper?") == "chat"
    assert detect_intent("Explain this research") == "chat"


# Test case-insensitive detection
def test_case_insensitive_detection():
    assert detect_intent("SUMMARIZE THIS PAPER") == "summary"
    assert detect_intent("GENERATE VIVA QUESTIONS") == "viva"


# Test agent routing
def test_agent_routing():
    assert route_intent("Summarize the paper") == "Summarizer Agent"
    assert route_intent("Explain the algorithms") == "Concept Extraction Agent"
    assert route_intent("Generate viva questions") == "Viva Agent"
    assert route_intent("Explain the main contribution") == "RAG Question Answering"
