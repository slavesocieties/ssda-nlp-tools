from run_pipeline import machine_stats


def test_machine_stats_preserve_full_review_count_without_claiming_full_queue():
    result = {"stats": {"review_pairs": 1_245_802, "identities": 32_518}}
    stats = machine_stats(result, displayed_review_pairs=5_000)
    assert stats["review_pairs"] == 1_245_802
    assert stats["review_pairs_displayed"] == 5_000
    assert stats["review_pairs_persisted_in_full"] is False
