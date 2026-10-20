from evals.repair_memory_evaluation import evaluate


def test_memory_controls_are_explicitly_authored_not_real_model_performance():
    report = evaluate()
    assert report["coding_resolution_measured"] is False
    assert report["provider_calls"] == 0
    assert report["passed"] == report["total"] == 11
