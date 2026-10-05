from sentence_transformers import CrossEncoder

from app.core.config import settings


class RerankingService:
    """
    Reranks retrieved document chunks using a cross-encoder.

    The cross-encoder receives the query and each candidate chunk
    together and produces a relevance score.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not candidates:
            return []

        pairs = [
            [query, candidate["content"]]
            for candidate in candidates
        ]

        scores = self.model.predict(pairs)

        ranked_candidates = []

        for candidate, score in zip(candidates, scores):
            ranked_candidates.append(
                {
                    **candidate,
                    "rerank_score": float(score),
                }
            )

        ranked_candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked_candidates[:top_k]