
from core.intent_router import detect_intent, route_intent


test_requests = [
    "Summarize this research paper",
    "Explain the algorithms used",
    "Prepare viva questions for me",
    "What are the main findings?",
    "What is the significance of this paper?"
]


for request in test_requests:

    intent = detect_intent(request)
    agent = route_intent(request)

    print("\nUser Request:", request)
    print("Detected Intent:", intent)
    print("Selected Agent:", agent)
