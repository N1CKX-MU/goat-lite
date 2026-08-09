"""Canonical GOAT-Bench goal-object vocabulary.

Single source of truth for the object categories the finetuned detector is
trained to emit. Shared by the dataset builder, the trainer's data.yaml, the
detector, and the goal matcher so class ids/names stay consistent end-to-end.

The list is the set of goal categories that appear in HM3D val_unseen (36
categories). Training images are rendered from HM3D *train* scenes, but the
class set we detect must equal the categories we are evaluated on.
"""

from __future__ import annotations

# Sorted for a stable, reproducible id<->name mapping.
GOAT_CATEGORIES: list[str] = [
    "boiler",
    "book",
    "calendar",
    "carpet",
    "christmas tree",
    "decorative plant",
    "dresser",
    "exercise bike",
    "flower vase",
    "flowerpot",
    "footrest",
    "freezer",
    "glass",
    "handrail",
    "hanger",
    "hanging clothes",
    "island",
    "microwave",
    "mirror",
    "nightstand",
    "parapet",
    "photo",
    "photo mount",
    "piano",
    "picture",
    "pillow",
    "plant",
    "printer",
    "radiator",
    "refrigerator",
    "rug",
    "shower glass",
    "stair",
    "statue",
    "vase",
    "window glass",
]

NUM_CLASSES: int = len(GOAT_CATEGORIES)

ID_TO_NAME: dict[int, str] = dict(enumerate(GOAT_CATEGORIES))
NAME_TO_ID: dict[str, int] = {name: i for i, name in ID_TO_NAME.items()}


def normalize(name: str) -> str:
    """Normalize a raw category string for lookup (lowercase, collapse spaces)."""
    return " ".join(name.strip().lower().split())


def name_to_id(name: str) -> int | None:
    """Return the class id for a category name, or None if not a GOAT category.

    Exact match against the GOAT vocabulary. Use this wherever the emitted class
    name matters (matcher, detector output).
    """
    return NAME_TO_ID.get(normalize(name))


# HM3D-Semantics annotators use free-form strings, so a number of instances
# describe a GOAT category under a different surface form. Without these, whole
# classes lose their training data: a 12-scene census found "stairs" 65 times vs
# "stair" 5, and "island" appeared *only* as "kitchen island".
#
# Curated deliberately -- only strings denoting the same physical object. These
# near-misses were rejected as different objects: "book rack", "piano stool",
# "piano bench", "mirror frame", "refrigerator cabinet", "stair wall",
# "stair handle", "glasses" (eyewear, not a glass surface).
#
# Ingestion-side only. It never changes the 36 emitted class names.
HM3D_CATEGORY_ALIASES: dict[str, str] = {
    "stairs": "stair",
    "clothes hanger": "hanger",
    "cloth hanger": "hanger",
    "wall hanger": "hanger",
    "kitchen island": "island",
    "staircase handrail": "handrail",
    "cook book": "book",
    "fur carpet": "carpet",
    "exhibition picture": "picture",
    "stained glass": "glass",
}


def hm3d_name_to_id(name: str) -> int | None:
    """Map a raw HM3D-Semantics category string to a GOAT class id.

    Exact GOAT match first, then the curated alias table. Used when building the
    detector's training set from HM3D annotations.
    """
    key = normalize(name)
    direct = NAME_TO_ID.get(key)
    if direct is not None:
        return direct
    alias = HM3D_CATEGORY_ALIASES.get(key)
    return NAME_TO_ID.get(alias) if alias else None
