
from agents.viva_agent import viva_prompt

sample_content = """
The research paper investigates the perspectives of
faculty members and students regarding artificial
intelligence in higher education.

Students commonly use AI for summarization,
idea generation and writing assistance.

Faculty members express concerns about academic
integrity, plagiarism and critical thinking.

The paper highlights the need for institutional
policies and AI awareness programs.
"""

messages = viva_prompt.format_messages(
    paper_content=sample_content
)

print("Viva Agent prompt created successfully!")

for message in messages:
    print("\nRole:", message.type)
    print(message.content)
