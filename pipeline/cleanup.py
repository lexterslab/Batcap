"""
Cleanup — bereinigt den zusammengeführten Tag-String.
Repliziert die Regex-Kette aus dem Workflow (Nodes 5342, 5341, 5371).
"""
import re
import logging

logger = logging.getLogger(__name__)


def clean_tags(raw_tags: str) -> str:
    """
    Führt die vollständige Bereinigungskette aus:
      1. Underscores → Leerzeichen  (Node 5371: _  →  )
      2. Klammern entfernen          (Node 5341: (\(|\)) → '')
      3. Artwork-Suffix entfernen    (Node 5345: ,\s\w+_\(artwork\) → '')
      4. Normalisierung              (führende/nachfolgende Kommas, doppelte Leerzeichen)
    """
    tags = raw_tags

    # 1. Underscores durch Leerzeichen ersetzen
    tags = tags.replace("_", " ")

    # 2. Alle Klammern ( ) entfernen
    tags = re.sub(r"[()]", "", tags)

    # 3. Artwork-Suffixe entfernen, z.B. ", digital artwork" oder ", ink_(artwork)"
    tags = re.sub(r",\s*\w[\w ]*_?\(?artwork\)?", "", tags, flags=re.IGNORECASE)

    # 4. Mehrfache Kommas und Leerzeichen normalisieren
    tags = re.sub(r",\s*,+", ",", tags)          # doppelte Kommas
    tags = re.sub(r"\s{2,}", " ", tags)           # mehrfache Leerzeichen
    tags = re.sub(r"^\s*,\s*", "", tags)          # führendes Komma
    tags = re.sub(r",\s*$", "", tags)             # nachfolgendes Komma

    # 5. Einheitliches Format: "tag1, tag2, tag3"
    parts = [t.strip() for t in tags.split(",") if t.strip()]
    result = ", ".join(parts)

    logger.debug(f"Tags nach Cleanup: {len(parts)} Tags")
    return result
