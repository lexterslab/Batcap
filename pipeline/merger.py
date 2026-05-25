"""
Merger — 1:1-Port der TagNormalizeCombine-Logik aus SamplingUtils von Silveroxides.

Normalisiert die Scores jedes Modells auf den Bereich [0.000001, 0.999999],
führt alle Tag-Listen zusammen, dedupliziert (höchster Score gewinnt)
und sortiert absteigend nach Score.
"""
from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hilfsfunktionen (identisch zur Original-Implementierung)
# ---------------------------------------------------------------------------

def _normalize_scores(scores: dict) -> dict:
    """Normalisiert Score-Werte auf [0.000001, 0.999999]."""
    if not scores:
        return {}
    vals    = [float(v) for v in scores.values()]
    vmin    = min(vals)
    vmax    = max(vals)
    lo, hi  = 0.000001, 0.999999
    if vmax == vmin:
        return {k: hi for k in scores}
    return {
        k: lo + (float(v) - vmin) / (vmax - vmin) * (hi - lo)
        for k, v in scores.items()
    }


def _even_scores(tags: list[str]) -> dict:
    """Erzeugt gleichmäßig verteilte Scores wenn keine echten Scores vorhanden sind."""
    n = len(tags)
    if n == 0:
        return {}
    if n == 1:
        return {tags[0]: 0.999999}
    lo, hi = 0.000001, 0.999999
    return {
        tag: hi - i * (hi - lo) / (n - 1)
        for i, tag in enumerate(tags)
    }


def _parse_tags(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    if not raw or not isinstance(raw, str):
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_scores(raw: dict | str | None) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


# ---------------------------------------------------------------------------
# Kern-Merge-Funktion (zwei Sätze)
# ---------------------------------------------------------------------------

def merge_two(
    tags1: str | list,
    scores1: dict | str | None,
    tags2: str | list,
    scores2: dict | str | None,
) -> tuple[str, dict]:
    """
    Führt zwei Tag-Sätze zusammen.
    Gibt (deduped_tags_str, normalized_scores_dict) zurück.
    """
    t1 = _parse_tags(tags1)
    t2 = _parse_tags(tags2)
    s1 = _parse_scores(scores1)
    s2 = _parse_scores(scores2)

    n1 = _normalize_scores(s1) if s1 else _even_scores(t1)
    n2 = _normalize_scores(s2) if s2 else _even_scores(t2)

    combined: dict[str, float] = {}

    for tag in t1:
        combined[tag] = n1.get(tag, 0.000001)

    for tag in t2:
        score = n2.get(tag, 0.000001)
        if tag in combined:
            if score > combined[tag]:
                combined[tag] = score
        else:
            combined[tag] = score

    sorted_tags = sorted(combined, key=lambda x: combined[x], reverse=True)
    return (
        ", ".join(sorted_tags),
        {t: combined[t] for t in sorted_tags},
    )


# ---------------------------------------------------------------------------
# Alle drei Modelle zusammenführen (Workflow-Reihenfolge: JTP2+JTP3 → +DINOv3)
# ---------------------------------------------------------------------------

def merge_all(
    jtp2_tags:   str, jtp2_scores:  dict,
    jtp3_tags:   str, jtp3_scores:  dict,
    dino_tags:   str, dino_scores:  dict,
) -> str:
    """
    Gibt den finalen, deduplizierten und nach Score sortierten Tag-String zurück.
    (Noch mit Underscores — Cleanup erfolgt in cleanup.py)
    """
    # Schritt 1: JTP PILOT v2 + JTP-3 Hydra
    merged12, scores12 = merge_two(jtp2_tags, jtp2_scores, jtp3_tags, jtp3_scores)
    logger.debug(f"Nach JTP2+JTP3 Merge: {len(merged12.split(','))} Tags")

    # Schritt 2: (JTP2+JTP3) + DINOv3
    merged_all, _ = merge_two(merged12, scores12, dino_tags, dino_scores)
    logger.debug(f"Nach finalem Merge: {len(merged_all.split(','))} Tags")

    return merged_all
