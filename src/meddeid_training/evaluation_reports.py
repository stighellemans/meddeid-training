"""Dependency-light, auditable evaluation slices for token classifiers."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any, Iterable, Mapping, Sequence


def _span_key(span: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(span["begin"]), int(span["end"]), str(span["label"])


def _exact_metrics(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    indices: Iterable[int],
    *,
    label: str | None = None,
) -> dict[str, int | float]:
    true = predicted = correct = documents = 0
    for index in indices:
        documents += 1
        gold_keys = {
            _span_key(span)
            for span in records[index].get("spans", [])
            if label is None or str(span.get("label")) == label
        }
        predicted_keys = {
            _span_key(span)
            for span in predictions[index]
            if label is None or str(span.get("label")) == label
        }
        true += len(gold_keys)
        predicted += len(predicted_keys)
        correct += len(gold_keys & predicted_keys)
    precision = correct / predicted if predicted else 0.0
    recall = correct / true if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "documents": documents,
        "true": true,
        "predicted": predicted,
        "correct": correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _metadata(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _profile(record: Mapping[str, Any]) -> str:
    metadata = _metadata(record)
    value = metadata.get("lang") or metadata.get("generation_profile") or "unknown"
    return str(value).split("@", 1)[0].replace("_", "-")


def _format_style(record: Mapping[str, Any]) -> str:
    style = _metadata(record).get("style_profile")
    if isinstance(style, Mapping):
        return str(style.get("name") or "unknown")
    return str(style or "unknown")


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _is_pediatric(record: Mapping[str, Any]) -> bool:
    metadata = _metadata(record)
    patient = metadata.get("patient")
    if not isinstance(patient, Mapping):
        return False
    born = _parse_date(patient.get("birth_date"))
    encounter = _parse_date(metadata.get("document_creation_date"))
    if born is None or encounter is None:
        return False
    years = encounter.year - born.year - ((encounter.month, encounter.day) < (born.month, born.day))
    return years < 18


def _overlap(left: tuple[int, int, str], right: tuple[int, int, str]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _boundary_report(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    counts = {
        "exact": 0,
        "boundary_error_same_label": 0,
        "label_error_same_boundary": 0,
        "overlap_wrong_label_and_boundary": 0,
        "missed_without_overlap": 0,
        "spurious_without_overlap": 0,
    }
    for record, predicted_spans in zip(records, predictions, strict=True):
        gold = [_span_key(span) for span in record.get("spans", [])]
        predicted = [_span_key(span) for span in predicted_spans]
        predicted_set = set(predicted)
        gold_set = set(gold)
        counts["exact"] += len(gold_set & predicted_set)
        for item in gold:
            if item in predicted_set:
                continue
            if any(item[:2] == candidate[:2] for candidate in predicted):
                counts["label_error_same_boundary"] += 1
            elif any(item[2] == candidate[2] and _overlap(item, candidate) for candidate in predicted):
                counts["boundary_error_same_label"] += 1
            elif any(_overlap(item, candidate) for candidate in predicted):
                counts["overlap_wrong_label_and_boundary"] += 1
            else:
                counts["missed_without_overlap"] += 1
        for item in predicted:
            if item not in gold_set and not any(_overlap(item, candidate) for candidate in gold):
                counts["spurious_without_overlap"] += 1
    return counts


def _hard_negative_report(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"targets": 0, "rendered_occurrences": 0, "false_positive_occurrences": 0}
    )
    for record, predicted_spans in zip(records, predictions, strict=True):
        text = str(record.get("text") or "")
        predicted = [_span_key(span) for span in predicted_spans]
        for target in _metadata(record).get("hard_negative_targets", []):
            if not isinstance(target, Mapping):
                continue
            category = str(target.get("category") or "unknown")
            value = str(target.get("value") or "")
            categories[category]["targets"] += 1
            if not value:
                continue
            cursor = 0
            while True:
                begin = text.find(value, cursor)
                if begin < 0:
                    break
                end = begin + len(value)
                categories[category]["rendered_occurrences"] += 1
                if any(begin < span_end and span_begin < end for span_begin, span_end, _ in predicted):
                    categories[category]["false_positive_occurrences"] += 1
                cursor = max(end, begin + 1)
    rendered = sum(row["rendered_occurrences"] for row in categories.values())
    false_positives = sum(row["false_positive_occurrences"] for row in categories.values())
    return {
        "rendered_occurrences": rendered,
        "false_positive_occurrences": false_positives,
        "false_positive_rate": false_positives / rendered if rendered else 0.0,
        "by_category": {key: value for key, value in sorted(categories.items())},
    }


def build_evaluation_slice_report(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return exact-span and error reports for the standard training protocol."""

    if len(records) != len(predictions):
        raise ValueError("records and predictions must contain the same number of documents")
    all_indices = list(range(len(records)))
    locale_indices: dict[str, list[int]] = defaultdict(list)
    style_indices: dict[str, list[int]] = defaultdict(list)
    pediatric_indices: dict[str, list[int]] = defaultdict(list)
    labels: set[str] = set()
    for index, record in enumerate(records):
        profile = _profile(record)
        locale_indices[profile].append(index)
        style_indices[_format_style(record)].append(index)
        if _is_pediatric(record):
            pediatric_indices[profile].append(index)
        labels.update(str(span.get("label")) for span in record.get("spans", []))
        labels.update(str(span.get("label")) for span in predictions[index])

    return {
        "contract": "meddeid.training-evaluation-slices.v1",
        "overall": _exact_metrics(records, predictions, all_indices),
        "by_locale": {
            key: _exact_metrics(records, predictions, indices)
            for key, indices in sorted(locale_indices.items())
        },
        "by_label": {
            label: _exact_metrics(records, predictions, all_indices, label=label)
            for label in sorted(labels)
        },
        "by_format_style": {
            key: _exact_metrics(records, predictions, indices)
            for key, indices in sorted(style_indices.items())
        },
        "pediatric": {
            "overall": _exact_metrics(
                records,
                predictions,
                [index for indices in pediatric_indices.values() for index in indices],
            ),
            "by_locale": {
                key: _exact_metrics(records, predictions, indices)
                for key, indices in sorted(pediatric_indices.items())
            },
        },
        "boundary": _boundary_report(records, predictions),
        "hard_negatives": _hard_negative_report(records, predictions),
    }


__all__ = ["build_evaluation_slice_report"]
