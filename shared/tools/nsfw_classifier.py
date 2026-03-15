# AI-persona-framework
# Copyright (C) 2025 Kenneth Haider
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
NSFW Output Classifier (Safety Layer 2)
----------------------------------------
Post-generation image classifier using Falconsai/nsfw_image_detection (ViT-based).

This is a safety net, not the primary gate. Its main purpose is catching
unexpected NSFW content from safe-looking prompts (SD models can do this).

Behavior by user permission level:
    - adult: classify and tag, never block
    - teen/child: classify and block if NSFW score exceeds threshold
"""

import os
from pathlib import Path
from typing import Optional

from PIL import Image

import logging

log = logging.getLogger(__name__)

_classifier = None
_MODEL_NAME = "Falconsai/nsfw_image_detection"

# NSFW score threshold for blocking (non-adult users only)
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.5"))


def _get_classifier():
    """Lazy-load the classifier pipeline (CPU-only, ~100MB model)."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "image-classification",
            model=_MODEL_NAME,
            device="cpu",
        )
    return _classifier


def classify_image(image_path: str) -> dict:
    """
    Classify a single image for NSFW content.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        {"nsfw_score": float, "label": "normal"|"nsfw", "blocked": False}
        The caller decides whether to act on blocked based on user_permission.
    """
    clf = _get_classifier()
    img = Image.open(image_path).convert("RGB")
    results = clf(img)

    scores = {r["label"]: r["score"] for r in results}
    nsfw_score = scores.get("nsfw", 0.0)
    label = "nsfw" if nsfw_score >= NSFW_THRESHOLD else "normal"

    return {
        "nsfw_score": nsfw_score,
        "label": label,
    }


def check_output(image_path: str, user_permission: str) -> dict:
    """
    Safety Layer 2: classify image and decide whether to block.

    For adult users, never blocks — just returns the classification.
    For teen/child users, blocks if NSFW score exceeds threshold.

    Args:
        image_path: Absolute path to the generated image.
        user_permission: "adult", "teen", or "child"

    Returns:
        {
            "nsfw_score": float,
            "label": "normal"|"nsfw",
            "blocked": bool,
            "reason": str|None,
        }
    """
    classification = classify_image(image_path)

    blocked = False
    reason = None

    if user_permission != "adult" and classification["label"] == "nsfw":
        blocked = True
        reason = (
            f"Generated image classified as NSFW "
            f"(score: {classification['nsfw_score']:.2f}, "
            f"threshold: {NSFW_THRESHOLD})"
        )
        # Delete the offending image
        try:
            os.remove(image_path)
        except OSError as e:
            log.warning("Failed to delete blocked NSFW image %s: %s", image_path, e)

    return {
        "nsfw_score": classification["nsfw_score"],
        "label": classification["label"],
        "blocked": blocked,
        "reason": reason,
    }
