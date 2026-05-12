import os

import google.generativeai as genai

from dotenv import load_dotenv

from app.rag.rag_tool import rag_query

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)


def run_crew(query: str):

    results = rag_query(query)

    if not results:

        return {
            "answer": (
                "Reliable medical answer "
                "not found in database.\n\n"
                "This is not medical advice."
            ),
            "sources": []
        }

    context = "\n\n".join(
        [r["chunk"] for r in results]
    )

    prompt = f"""
You are a medical retrieval assistant.

Answer ONLY using the retrieved
medical context below.

If the context is incomplete,
weak, or unrelated, say:

"Reliable medical answer not found in database."

Rules:
- Do NOT hallucinate
- Do NOT guess
- Do NOT use outside knowledge
- Give concise answer
- Mention evidence briefly

Medical Context:
{context}

Question:
{query}

Final Answer:
"""

    try:

        response = model.generate_content(
            prompt
        )

        answer = response.text.strip()

    except Exception as e:

        answer = f"Generation Error: {str(e)}"

    answer += "\n\nThis is not medical advice."

    return {
        "answer": answer,
        "sources": [
            r["chunk"] for r in results
        ]
    }
