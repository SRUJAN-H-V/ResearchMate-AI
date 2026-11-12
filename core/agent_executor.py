
from core.intent_router import detect_intent
from agents.summarizer import summarize_paper
from agents.concept_extractor import extract_concepts
from agents.viva_agent import generate_viva_questions
from core.rag_chain import answer_question


def execute_agent(user_input, paper_content, vector_store):

    # Step 1: Detect user intent
    intent = detect_intent(user_input)

    # Step 2: Execute the appropriate agent
    if intent == "summary":
        response = summarize_paper(paper_content)

    elif intent == "concept":
        response = extract_concepts(paper_content)

    elif intent == "viva":
        response = generate_viva_questions(paper_content)

    else:
        response = answer_question(
            vector_store,
            user_input
        )

    return {
        "intent": intent,
        "response": response
    }
