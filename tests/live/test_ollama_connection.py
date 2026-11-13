from langchain_ollama import ChatOllama

# Connect to the local Qwen model
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)

# Send a test prompt
response = llm.invoke(
    "Explain RAG in simple words in 3 sentences."
)

print("\nOllama Response:")
print(response.content)