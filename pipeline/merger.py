"""
Merger — Normalisiert und führt Tag-Listen aller aktiven Tagger zusammen.
1:1-Port der TagNormalizeCombine-Logik aus SamplingUtils von Silveroxides.
Unterstützt beliebig viele Tagger-Eingaben.
"""
from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _normalize_scores(scores: dict) -> dict:
    if not scores:
        return {}
    vals   = [float(v) for v in scores.values()]
    vmin, vmax = min(vals), max(vals)
    lo, hi = 0.000001, 0.999999
    if vmax == vmin:
        return {k: hi for k in scores}
    return {k: lo + (float(v) - vmin) / (vmax - vmin) * (hi - lo)
            for k, v in scores.items()}


def _even_scores(tags: list[str]) -> dict:
    n = len(tags)
    if n == 0: return {}
    if n == 1: return {tags[0]: 0.999999}
    lo, hi = 0.000001, 0.999999
    return {tag: hi - i * (hi - lo) / (n - 1) for i, tag in enumerate(tags)}


def _parse_tags(raw: str | list) -> list[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    if not raw or not isinstance(raw, str):
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_scores(raw: dict | str | None) -> dict:
    if raw is None:       return {}
    if isinstance(raw, dict): return raw
    if isinstance(raw, str):
        try:    return json.loads(raw)
        except: return {}
    return {}


# ── Kern-Merge (zwei Sätze) ───────────────────────────────────────────────────

def merge_two(
    tags1: str | list, scores1: dict | str | None,
    tags2: str | list, scores2: dict | str | None,
) -> tuple[str, dict]:
    t1, t2 = _parse_tags(tags1), _parse_tags(tags2)
    s1, s2 = _parse_scores(scores1), _parse_scores(scores2)

    n1 = _normalize_scores(s1) if s1 else _even_scores(t1)
    n2 = _normalize_scores(s2) if s2 else _even_scores(t2)

    combined: dict[str, float] = {}
    for tag in t1:
        combined[tag] = n1.get(tag, 0.000001)
    for tag in t2:
        score = n2.get(tag, 0.000001)
        combined[tag] = max(combined.get(tag, 0.0), score)

    sorted_tags = sorted(combined, key=lambda x: combined[x], reverse=True)
    return ", ".join(sorted_tags), {t: combined[t] for t in sorted_tags}


# ── Alle Tagger zusammenführen ────────────────────────────────────────────────

def merge_all(results: dict[str, tuple[str, dict]]) -> str:
    """
    Nimmt das dict aus tagger.run_all() und gibt den finalen Tag-String zurück.
    Leere Einträge (deaktivierte Tagger) werden übersprungen.
    """
    # Nur aktive Tagger (nicht-leere Tags)
    active = [(tags, scores) for tags, scores in results.values() if tags.strip()]

    if not active:
        logger.warning("Alle Tagger deaktiviert oder keine Tags zurückgegeben.")
        return ""

    # Iterativ mergen: (acc_tags, acc_scores) + next
    acc_tags, acc_scores = active[0]
    for tags, scores in active[1:]:
        acc_tags, acc_scores = merge_two(acc_tags, acc_scores, tags, scores)

    logger.debug("Merge-Ergebnis: %d Tags", len(acc_tags.split(",")))
    return acc_tags
