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
    """Return the class id for a category name, or None if not a GOAT category."""
    return NAME_TO_ID.get(normalize(name))
