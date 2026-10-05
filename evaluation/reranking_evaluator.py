import json

from app.db.session import SessionLocal
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import EmbeddingService
from app.services.reranking_service import RerankingService


DATASET_PATH = "evaluation/retrieval_dataset.json"

CANDIDATE_TOP_K = 10
FINAL_TOP_K = 3


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def expected_filenames(item):
    return {
        source["filename"]
        for source in item["expected_sources"]
    }


def calculate_reciprocal_rank(results, expected):
    for rank, result in enumerate(results, start=1):
        if result["filename"] in expected:
            return 1 / rank

    return 0.0


def calculate_positive_metrics(results, expected):
    top_1 = results[:1]
    top_3 = results[:FINAL_TOP_K]

    recall_at_1 = any(
        result["filename"] in expected
        for result in top_1
    )

    recall_at_3 = any(
        result["filename"] in expected
        for result in top_3
    )

    relevant_at_3 = sum(
        result["filename"] in expected
        for result in top_3
    )

    precision_at_3 = relevant_at_3 / FINAL_TOP_K

    reciprocal_rank = calculate_reciprocal_rank(
        top_3,
        expected,
    )

    return {
        "recall_at_1": int(recall_at_1),
        "recall_at_3": int(recall_at_3),
        "precision_at_3": precision_at_3,
        "mrr_at_3": reciprocal_rank,
    }


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def classification_accuracy(
    positive_scores,
    negative_scores,
    threshold,
    higher_is_supported=True,
):
    correct = 0
    total = len(positive_scores) + len(negative_scores)

    if total == 0:
        return 0.0

    for score in positive_scores:
        if higher_is_supported:
            predicted_supported = score >= threshold
        else:
            predicted_supported = score <= threshold

        if predicted_supported:
            correct += 1

    for score in negative_scores:
        if higher_is_supported:
            predicted_supported = score >= threshold
        else:
            predicted_supported = score <= threshold

        if not predicted_supported:
            correct += 1

    return correct / total


def threshold_metrics(
    positive_scores,
    negative_scores,
    threshold,
    higher_is_supported=True,
):
    positive_total = len(positive_scores)
    negative_total = len(negative_scores)

    positive_correct = 0
    negative_correct = 0

    for score in positive_scores:
        if higher_is_supported:
            supported = score >= threshold
        else:
            supported = score <= threshold

        if supported:
            positive_correct += 1

    for score in negative_scores:
        if higher_is_supported:
            supported = score >= threshold
        else:
            supported = score <= threshold

        if not supported:
            negative_correct += 1

    positive_recall = (
        positive_correct / positive_total
        if positive_total
        else 0.0
    )

    negative_accuracy = (
        negative_correct / negative_total
        if negative_total
        else 0.0
    )

    false_positive_rate = (
        1 - negative_accuracy
    )

    accuracy = (
        (positive_correct + negative_correct)
        / (positive_total + negative_total)
        if positive_total + negative_total
        else 0.0
    )

    return {
        "threshold": threshold,
        "positive_recall": positive_recall,
        "negative_accuracy": negative_accuracy,
        "false_positive_rate": false_positive_rate,
        "accuracy": accuracy,
    }


def generate_candidate_thresholds(
    positive_scores,
    negative_scores,
):
    all_scores = sorted(
        set(
            positive_scores
            + negative_scores
        )
    )

    if not all_scores:
        return []

    thresholds = []

    # Slightly below the minimum score.
    thresholds.append(
        all_scores[0] - 0.001
    )

    # Midpoints between observed scores.
    for left, right in zip(
        all_scores,
        all_scores[1:],
    ):
        thresholds.append(
            (left + right) / 2
        )

    # Slightly above the maximum score.
    thresholds.append(
        all_scores[-1] + 0.001
    )

    return thresholds


def print_score_distribution(
    name,
    positive_scores,
    negative_scores,
):
    print("\n")
    print("=" * 60)
    print(f"{name} SCORE DISTRIBUTION")
    print("=" * 60)

    print(
        f"Positive queries: "
        f"{len(positive_scores)}"
    )

    if positive_scores:
        print(
            f"  min:     {min(positive_scores):.4f}"
        )
        print(
            f"  max:     {max(positive_scores):.4f}"
        )
        print(
            f"  average: {average(positive_scores):.4f}"
        )

    print(
        f"Negative queries: "
        f"{len(negative_scores)}"
    )

    if negative_scores:
        print(
            f"  min:     {min(negative_scores):.4f}"
        )
        print(
            f"  max:     {max(negative_scores):.4f}"
        )
        print(
            f"  average: {average(negative_scores):.4f}"
        )


def print_best_thresholds(
    name,
    positive_scores,
    negative_scores,
    higher_is_supported,
):
    thresholds = generate_candidate_thresholds(
        positive_scores,
        negative_scores,
    )

    results = []

    for threshold in thresholds:
        metrics = threshold_metrics(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            threshold=threshold,
            higher_is_supported=higher_is_supported,
        )

        results.append(metrics)

    results.sort(
        key=lambda item: (
            item["accuracy"],
            item["negative_accuracy"],
            item["positive_recall"],
        ),
        reverse=True,
    )

    print("\n")
    print("=" * 60)
    print(f"{name} THRESHOLD ANALYSIS")
    print("=" * 60)

    print(
        f"{'Threshold':>12} "
        f"{'Pos Recall':>12} "
        f"{'Neg Acc':>12} "
        f"{'FPR':>12} "
        f"{'Accuracy':>12}"
    )

    print("-" * 60)

    for result in results[:10]:
        print(
            f"{result['threshold']:>12.4f} "
            f"{result['positive_recall']:>12.2f} "
            f"{result['negative_accuracy']:>12.2f} "
            f"{result['false_positive_rate']:>12.2f} "
            f"{result['accuracy']:>12.2f}"
        )

    if results:
        best = results[0]

        print("\nBest threshold by overall accuracy:")

        print(
            f"Threshold:          "
            f"{best['threshold']:.4f}"
        )

        print(
            f"Positive Recall:    "
            f"{best['positive_recall']:.2f}"
        )

        print(
            f"Negative Accuracy:  "
            f"{best['negative_accuracy']:.2f}"
        )

        print(
            f"False Positive Rate: "
            f"{best['false_positive_rate']:.2f}"
        )

        print(
            f"Overall Accuracy:    "
            f"{best['accuracy']:.2f}"
        )

        return best

    return None


def main():
    dataset = load_dataset()

    db = SessionLocal()

    try:
        repository = DocumentRepository(db)
        embedding_service = EmbeddingService()
        reranking_service = RerankingService()

        dense_positive_distances = []
        dense_negative_distances = []

        reranker_positive_scores = []
        reranker_negative_scores = []

        baseline_positive_metrics = []
        reranked_positive_metrics = []

        print("\n")
        print("=" * 60)
        print("RERANKER ANSWERABILITY CALIBRATION")
        print("=" * 60)

        for item in dataset:
            query = item["query"]
            expected = expected_filenames(item)

            query_embedding = embedding_service.embed_text(
                query
            )

            dense_results = repository.similarity_search(
                query_embedding=query_embedding,
                top_k=CANDIDATE_TOP_K,
            )

            candidates = []

            for chunk, distance in dense_results:
                candidates.append(
                    {
                        "document_id": chunk.document_id,
                        "filename": chunk.document.filename,
                        "chunk_id": chunk.id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "distance": float(distance),
                    }
                )

            baseline_results = candidates[:FINAL_TOP_K]

            reranked_results = reranking_service.rerank(
                query=query,
                candidates=candidates,
                top_k=FINAL_TOP_K,
            )

            # The strongest dense candidate has the lowest
            # cosine distance.
            if candidates:
                top_dense_distance = candidates[0]["distance"]
            else:
                top_dense_distance = None

            # The strongest reranker candidate has the highest
            # cross-encoder score.
            if reranked_results:
                top_reranker_score = reranked_results[0][
                    "rerank_score"
                ]
            else:
                top_reranker_score = None

            print("\nQuery:")
            print(query)

            print(
                f"Type: "
                f"{'POSITIVE' if expected else 'NEGATIVE'}"
            )

            print(
                f"Expected: "
                f"{sorted(expected)}"
            )

            print(
                f"Dense top distance: "
                f"{top_dense_distance:.4f}"
                if top_dense_distance is not None
                else "Dense top distance: N/A"
            )

            print(
                f"Reranker top score: "
                f"{top_reranker_score:.4f}"
                if top_reranker_score is not None
                else "Reranker top score: N/A"
            )

            print(
                "Baseline: "
                f"{[r['filename'] for r in baseline_results]}"
            )

            print(
                "Reranked: "
                f"{[r['filename'] for r in reranked_results]}"
            )

            if expected:
                if top_dense_distance is not None:
                    dense_positive_distances.append(
                        top_dense_distance
                    )

                if top_reranker_score is not None:
                    reranker_positive_scores.append(
                        top_reranker_score
                    )

                baseline_positive_metrics.append(
                    calculate_positive_metrics(
                        baseline_results,
                        expected,
                    )
                )

                reranked_positive_metrics.append(
                    calculate_positive_metrics(
                        reranked_results,
                        expected,
                    )
                )

            else:
                if top_dense_distance is not None:
                    dense_negative_distances.append(
                        top_dense_distance
                    )

                if top_reranker_score is not None:
                    reranker_negative_scores.append(
                        top_reranker_score
                    )

        # =============================================
        # Ranking metrics
        # =============================================

        print("\n")
        print("=" * 60)
        print("POSITIVE QUERY RANKING METRICS")
        print("=" * 60)

        for metric in [
            "recall_at_1",
            "recall_at_3",
            "precision_at_3",
            "mrr_at_3",
        ]:
            baseline_value = average(
                [
                    result[metric]
                    for result in baseline_positive_metrics
                ]
            )

            reranked_value = average(
                [
                    result[metric]
                    for result in reranked_positive_metrics
                ]
            )

            print(
                f"{metric:<18}"
                f"baseline={baseline_value:.2f} "
                f"reranked={reranked_value:.2f} "
                f"change="
                f"{reranked_value - baseline_value:+.2f}"
            )

        # =============================================
        # Score distributions
        # =============================================

        print_score_distribution(
            "DENSE COSINE DISTANCE",
            dense_positive_distances,
            dense_negative_distances,
        )

        print_score_distribution(
            "RERANKER CROSS-ENCODER SCORE",
            reranker_positive_scores,
            reranker_negative_scores,
        )

        # =============================================
        # Dense threshold analysis
        # Lower distance = more relevant
        # =============================================

        dense_best = print_best_thresholds(
            name="DENSE DISTANCE",
            positive_scores=dense_positive_distances,
            negative_scores=dense_negative_distances,
            higher_is_supported=False,
        )

        # =============================================
        # Reranker threshold analysis
        # Higher score = more relevant
        # =============================================

        reranker_best = print_best_thresholds(
            name="RERANKER SCORE",
            positive_scores=reranker_positive_scores,
            negative_scores=reranker_negative_scores,
            higher_is_supported=True,
        )

        # =============================================
        # Final comparison
        # =============================================

        print("\n")
        print("=" * 60)
        print("ANSWERABILITY COMPARISON")
        print("=" * 60)

        if dense_best:
            print(
                "Dense retrieval best threshold:"
            )

            print(
                f"  threshold = "
                f"{dense_best['threshold']:.4f}"
            )

            print(
                f"  positive recall = "
                f"{dense_best['positive_recall']:.2f}"
            )

            print(
                f"  negative accuracy = "
                f"{dense_best['negative_accuracy']:.2f}"
            )

            print(
                f"  overall accuracy = "
                f"{dense_best['accuracy']:.2f}"
            )

        if reranker_best:
            print(
                "\nReranker best threshold:"
            )

            print(
                f"  threshold = "
                f"{reranker_best['threshold']:.4f}"
            )

            print(
                f"  positive recall = "
                f"{reranker_best['positive_recall']:.2f}"
            )

            print(
                f"  negative accuracy = "
                f"{reranker_best['negative_accuracy']:.2f}"
            )

            print(
                f"  overall accuracy = "
                f"{reranker_best['accuracy']:.2f}"
            )

        if dense_best and reranker_best:
            print("\nInterpretation:")

            if (
                reranker_best["accuracy"]
                > dense_best["accuracy"]
            ):
                print(
                    "The reranker provides better "
                    "answerability discrimination "
                    "on this evaluation dataset."
                )

            elif (
                reranker_best["accuracy"]
                < dense_best["accuracy"]
            ):
                print(
                    "Dense similarity provides better "
                    "answerability discrimination "
                    "on this evaluation dataset."
                )

            else:
                print(
                    "The reranker and dense similarity "
                    "provide the same best classification "
                    "accuracy on this evaluation dataset."
                )

            print(
                "\nImportant: the reported thresholds "
                "are dataset-calibrated evaluation "
                "thresholds. They should not automatically "
                "be promoted to production configuration."
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()

