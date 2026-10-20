from evals.execution_retrieval_evaluation import evaluate


def test_multilanguage_localization_controls_are_explicitly_not_resolution_scores():
    report = evaluate()
    assert report["coding_resolution_measured"] is False
    assert report["provider_calls"] == 0
    assert report["total"] == 6
    assert report["execution_file_recall"] == 1.0
    assert all(row["execution_estimated_tokens"] <= 1024 for row in report["rows"])
