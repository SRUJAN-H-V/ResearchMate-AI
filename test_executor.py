
from unittest.mock import patch
from core.agent_executor import execute_agent


# Sample paper content for testing
paper_content = "This is sample research paper content."

# Test different user requests
test_requests = [
    "Summarize this research paper",
    "Explain the algorithms used",
    "Prepare viva questions",
    "What is the significance of this paper?"
]

# Replace actual agent functions with mock functions.
# This prevents Gemini API calls during testing.
with patch("core.agent_executor.summarize_paper",
           return_value="Summary Agent executed successfully"):

    with patch("core.agent_executor.extract_concepts",
               return_value="Concept Agent executed successfully"):

        with patch("core.agent_executor.generate_viva_questions",
                   return_value="Viva Agent executed successfully"):

            with patch("core.agent_executor.answer_question",
                       return_value="RAG Agent executed successfully"):

                for request in test_requests:

                    result = execute_agent(
                        request,
                        paper_content,
                        vector_store=None
                    )

                    print("\nUser Request:", request)
                    print("Detected Intent:", result["intent"])
                    print("Agent Response:", result["response"])
