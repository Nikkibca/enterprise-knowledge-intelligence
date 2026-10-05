from app.core.config import settings
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.reranking_service import RerankingService


class RAGService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        reranking_service: RerankingService | None = None,
        similarity_threshold: float | None = None,
        reranker_threshold: float | None = None,
        retrieval_top_k: int | None = None,
    ):
        self.repository = repository
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.reranking_service = reranking_service

        self.similarity_threshold = (
            settings.retrieval_similarity_threshold
            if similarity_threshold is None
            else similarity_threshold
        )

        self.reranker_threshold = (
            settings.reranker_threshold
            if reranker_threshold is None
            else reranker_threshold
        )

        self.retrieval_top_k = (
            settings.retrieval_top_k
            if retrieval_top_k is None
            else retrieval_top_k
        )

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:
        query_embedding = self.embedding_service.embed_text(question)

        candidate_count = max(
            self.retrieval_top_k,
            top_k,
        )

        search_results = self.repository.similarity_search(
            query_embedding=query_embedding,
            top_k=candidate_count,
        )

        if not search_results:
            return {
                "question": question,
                "answer": (
                    "The information is not available "
                    "in the provided documents."
                ),
                "sources": [],
            }

        candidates = []

        for chunk, distance in search_results:
            candidates.append(
                {
                    "chunk": chunk,
                    "distance": distance,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "filename": chunk.document.filename,
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata_,
                }
            )

        if self.reranking_service is not None:
            reranked = self.reranking_service.rerank(
                query=question,
                candidates=candidates,
                top_k=candidate_count,
            )

            relevant_candidates = [
                candidate
                for candidate in reranked
                if candidate["rerank_score"]
                >= self.reranker_threshold
            ]

            relevant_candidates = relevant_candidates[:top_k]

        else:
            relevant_candidates = [
                candidate
                for candidate in candidates
                if candidate["distance"]
                <= self.similarity_threshold
            ][:top_k]

        if not relevant_candidates:
            return {
                "question": question,
                "answer": (
                    "The information is not available "
                    "in the provided documents."
                ),
                "sources": [],
            }

        context_parts = []

        for candidate in relevant_candidates:
            context_parts.append(
                f"Document: {candidate['filename']}\n"
                f"Content:\n{candidate['content']}"
            )

        context = "\n\n".join(context_parts)

        answer = self.llm_service.generate_answer(
            question=question,
            context=context,
        )

        sources = [
            {
                "document_id": candidate["document_id"],
                "filename": candidate["filename"],
                "chunk_id": candidate["chunk_id"],
                "chunk_index": candidate["chunk_index"],
                "content": candidate["content"],
                "distance": candidate["distance"],
                "metadata": candidate["metadata"],
            }
            for candidate in relevant_candidates
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }