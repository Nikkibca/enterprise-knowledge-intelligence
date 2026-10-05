from app.services.reranking_service import RerankingService


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []

        for query, content in pairs:
            if "reimbursement limit" in content.lower():
                scores.append(0.95)
            elif "vacation" in content.lower():
                scores.append(0.40)
            else:
                scores.append(0.10)

        return scores


def test_reranker_sorts_candidates_by_score():
    service = RerankingService.__new__(
        RerankingService
    )

    service.model = FakeCrossEncoder()

    candidates = [
        {
            "content": "Employees may take vacation days.",
            "document_id": 1,
        },
        {
            "content": "The employee reimbursement limit is $1,000.",
            "document_id": 2,
        },
        {
            "content": "The office is closed on public holidays.",
            "document_id": 3,
        },
    ]

    results = service.rerank(
        query="What is the employee reimbursement limit?",
        candidates=candidates,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0]["document_id"] == 2
    assert results[0]["rerank_score"] == 0.95

    assert results[1]["document_id"] == 1
    assert results[1]["rerank_score"] == 0.40