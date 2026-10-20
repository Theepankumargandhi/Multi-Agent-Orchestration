from evals.memory_snapshot_evaluation import evaluate


def test_snapshot_controls_remain_separate_from_real_model_quality():
    report = evaluate()
    assert report["passed"] == report["total"] == 8
    assert report["coding_resolution_measured"] is False
    assert report["semantic_independence_proven"] is False
    assert report["provider_calls"] == 0
