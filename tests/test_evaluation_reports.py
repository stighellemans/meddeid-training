from __future__ import annotations

from meddeid_training.evaluation_reports import build_evaluation_slice_report


def _span(text: str, value: str, label: str, *, begin: int | None = None) -> dict:
    start = text.index(value) if begin is None else begin
    return {"begin": start, "end": start + len(value), "text": value, "label": label}


def test_evaluation_report_includes_required_standard_slices() -> None:
    gb_text = "Patient Ada North reviewed at 08:30."
    us_text = "Patient Ben West reviewed in Boston."
    records = [
        {
            "document_id": "gb",
            "text": gb_text,
            "spans": [_span(gb_text, "Ada North", "Name:Patient")],
            "metadata": {
                "lang": "en-GB",
                "document_creation_date": "2025-01-01",
                "patient": {"birth_date": "2020-01-02"},
                "style_profile": {"name": "compact"},
                "hard_negative_targets": [{"category": "time", "value": "08:30"}],
            },
        },
        {
            "document_id": "us",
            "text": us_text,
            "spans": [
                _span(us_text, "Ben West", "Name:Patient"),
                _span(us_text, "Boston", "Address_Location:Other"),
            ],
            "metadata": {
                "lang": "en-US",
                "document_creation_date": "2025-01-01",
                "patient": {"birth_date": "1980-01-01"},
                "style_profile": {"name": "narrative"},
                "hard_negative_targets": [],
            },
        },
    ]
    predictions = [
        [
            _span(gb_text, "Ada North", "Name:Patient"),
            _span(gb_text, "08:30", "Date"),
        ],
        [
            # Same label, deliberately shortened boundary.
            {"begin": us_text.index("Ben West"), "end": us_text.index("Ben West") + 3, "text": "Ben", "label": "Name:Patient"},
            _span(us_text, "Boston", "Address_Location:Other"),
        ],
    ]

    report = build_evaluation_slice_report(records, predictions)

    assert report["contract"] == "meddeid.training-evaluation-slices.v1"
    assert set(report["by_locale"]) == {"en-GB", "en-US"}
    assert report["by_label"]["Address_Location:Other"]["f1"] == 1.0
    assert report["by_format_style"]["compact"]["documents"] == 1
    assert report["pediatric"]["by_locale"]["en-GB"]["documents"] == 1
    assert report["boundary"]["boundary_error_same_label"] == 1
    assert report["boundary"]["spurious_without_overlap"] == 1
    assert report["hard_negatives"]["false_positive_occurrences"] == 1
    assert report["hard_negatives"]["by_category"]["time"]["targets"] == 1
