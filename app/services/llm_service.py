import requests

from app.core.config import settings


class LLMService:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.llm_base_url

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        instructions = """
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information to answer,
   say that the information is not available in the provided documents.
4. Keep the answer concise and directly answer the question.
"""

        prompt = f"""
{instructions}

Context:

{context}

User question:

{question}

Answer:
"""

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        return response.json()["response"].strip()