import json

import requests


API_URL = "http://127.0.0.1:8000/search"

DATASET_PATH = "evaluation/retrieval_dataset.json"

NEGATIVE_DISTANCE_THRESHOLD = 0.7

TOP_K = 3


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def search(query: str, top_k: int = TOP_K):
    response = requests.post(
        API_URL,
        json={
            "query": query,
            "top_k": top_k,
        },
        timeout=60,
    )

    response.raise_for_status()

    return response.json()["results"]


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


def analyze_thresholds(dataset):
    """
    Analyze the trade-off between:

    1. Positive-query recall
       - How many genuinely relevant documents
         remain below a given distance threshold?

    2. Negative-query accuracy
       - How many unanswerable queries are correctly
         rejected because their nearest result is
         above the threshold?

    Lower cosine distance means greater similarity.
    """

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    positive_distances = []
    negative_distances = []

    for item in dataset:
        query = item["query"]

        results = search(
            query=query,
            top_k=TOP_K,
        )

        # -----------------------------------------
        # Positive / answerable query
        # -----------------------------------------

        if item["expected_sources"]:
            expected = expected_filenames(item)

            relevant_distances = [
                result["distance"]
                for result in results
                if result["filename"] in expected
            ]

            positive_distances.extend(
                relevant_distances
            )

        # -----------------------------------------
        # Negative / unanswerable query
        # -----------------------------------------

        else:
            if results:
                negative_distances.append(
                    results[0]["distance"]
                )

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    print(
        "\nPositive-query relevant distances:"
    )

    print(
        [
            round(distance, 4)
            for distance in positive_distances
        ]
    )

    print(
        "\nNegative-query top-1 distances:"
    )

    print(
        [
            round(distance, 4)
            for distance in negative_distances
        ]
    )

    print(
        "\nThreshold | Positive Recall | Negative Accuracy"
    )

    print("-" * 55)

    for threshold in thresholds:

        # For positive queries, the relevant result
        # is considered retrievable if its distance
        # is at or below the threshold.
        positive_recall = (
            sum(
                distance <= threshold
                for distance in positive_distances
            )
            / len(positive_distances)
            if positive_distances
            else 0.0
        )

        # For negative queries, we want the closest
        # result to be ABOVE the threshold.
        negative_accuracy = (
            sum(
                distance > threshold
                for distance in negative_distances
            )
            / len(negative_distances)
            if negative_distances
            else 0.0
        )

        print(
            f"{threshold:9.2f} | "
            f"{positive_recall:15.2f} | "
            f"{negative_accuracy:17.2f}"
        )


def main():
    dataset = load_dataset()

    positive_results = []
    negative_results = []

    print("\nRETRIEVAL EVALUATION\n")

    for item in dataset:
        query = item["query"]
        expected = expected_filenames(item)

        results = search(query)

        filenames = [
            result["filename"]
            for result in results
        ]

        distances = [
            result["distance"]
            for result in results
        ]

        print(f"Query: {query}")
        print(f"Expected: {sorted(expected)}")
        print(f"Retrieved: {filenames}")
        print(
            "Distances:",
            [round(distance, 4) for distance in distances],
        )

        # =========================================
        # Positive / answerable query
        # =========================================

        if expected:
            top_1 = results[:1]
            top_3 = results[:3]

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

            precision_at_3 = relevant_at_3 / 3

            reciprocal_rank = calculate_reciprocal_rank(
                top_3,
                expected,
            )

            positive_results.append(
                {
                    "recall_at_1": int(recall_at_1),
                    "recall_at_3": int(recall_at_3),
                    "precision_at_3": precision_at_3,
                    "mrr_at_3": reciprocal_rank,
                }
            )

        # =========================================
        # Negative / unanswerable query
        # =========================================

        else:
            has_relevant_candidate = any(
                result["distance"]
                <= NEGATIVE_DISTANCE_THRESHOLD
                for result in results
            )

            abstained = not has_relevant_candidate

            negative_results.append(
                {
                    "abstained": abstained,
                    "top_1_distance": (
                        results[0]["distance"]
                        if results
                        else None
                    ),
                }
            )

            print(
                "Abstention:",
                "PASS" if abstained else "FAIL",
            )

        print("-" * 60)

    # =========================================
    # Positive-query metrics
    # =========================================

    if positive_results:
        avg_recall_at_1 = sum(
            result["recall_at_1"]
            for result in positive_results
        ) / len(positive_results)

        avg_recall_at_3 = sum(
            result["recall_at_3"]
            for result in positive_results
        ) / len(positive_results)

        avg_precision_at_3 = sum(
            result["precision_at_3"]
            for result in positive_results
        ) / len(positive_results)

        mrr_at_3 = sum(
            result["mrr_at_3"]
            for result in positive_results
        ) / len(positive_results)

        print("\n" + "=" * 60)
        print("POSITIVE QUERY METRICS")
        print("=" * 60)

        print(
            f"Average Recall@1: "
            f"{avg_recall_at_1:.2f}"
        )

        print(
            f"Average Recall@3: "
            f"{avg_recall_at_3:.2f}"
        )

        print(
            f"Average Precision@3: "
            f"{avg_precision_at_3:.2f}"
        )

        print(
            f"MRR@3: "
            f"{mrr_at_3:.2f}"
        )

    # =========================================
    # Negative-query metrics
    # =========================================

    if negative_results:
        negative_accuracy = sum(
            result["abstained"]
            for result in negative_results
        ) / len(negative_results)

        false_positive_rate = 1 - negative_accuracy

        print("\n" + "=" * 60)
        print("NEGATIVE QUERY METRICS")
        print("=" * 60)

        print(
            f"Negative Query Accuracy: "
            f"{negative_accuracy:.2f}"
        )

        print(
            f"False Positive Rate: "
            f"{false_positive_rate:.2f}"
        )

        print(
            f"Negative-query threshold: "
            f"{NEGATIVE_DISTANCE_THRESHOLD:.2f}"
        )

    # =========================================
    # Threshold analysis
    # =========================================

    analyze_thresholds(dataset)


if __name__ == "__main__":
    main()