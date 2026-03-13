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
ComfyUI Workflow Templates
--------------------------
Pre-built workflow JSON templates for image generation.

Each template is a function that returns a ComfyUI API-format workflow dict,
parameterized by prompt, negative prompt, checkpoint, seed, etc.

Templates:
- sd15_basic: SD1.5 text-to-image (smallest VRAM, fastest, good for testing)
- sdxl_turbo: SDXL Turbo text-to-image (fast, no refiner needed)
- sd15_hires: SD1.5 with hi-res fix (better quality, slower)
"""

import random
from typing import Optional


def sd15_basic(
    prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, deformed",
    checkpoint: str = "SD1.5/AOM3A3_orangemixs.safetensors",
    width: int = 512,
    height: int = 512,
    steps: int = 25,
    cfg: float = 7.0,
    seed: Optional[int] = None,
) -> dict:
    """SD1.5 basic text-to-image workflow."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "persona_gen", "images": ["8", 0]},
        },
    }


def sdxl_turbo(
    prompt: str,
    negative_prompt: str = "ugly, blurry, low quality",
    checkpoint: str = "wildcardxXLTURBO_wildcardxXLTURBOV10.safetensors",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    cfg: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    """SDXL Turbo text-to-image workflow. Very fast (4 steps), low CFG."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "persona_gen", "images": ["8", 0]},
        },
    }


def sd15_nsfw(
    prompt: str,
    negative_prompt: str = "ugly, blurry, low quality, deformed",
    checkpoint: str = "SD1.5/AOM3A3_orangemixs.safetensors",
    width: int = 512,
    height: int = 512,
    steps: int = 30,
    cfg: float = 7.0,
    seed: Optional[int] = None,
) -> dict:
    """SD1.5 NSFW workflow — adult only. No NSFW suppression in negative prompt."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "persona_nsfw", "images": ["8", 0]},
        },
    }


def sdxl_safe(
    prompt: str,
    negative_prompt: str = (
        "ugly, blurry, low quality, deformed, nsfw, nude, naked, "
        "explicit, sexual, violence, gore, blood, weapons, drugs, "
        "horror, scary, disturbing"
    ),
    checkpoint: str = "wildcardxXLTURBO_wildcardxXLTURBOV10.safetensors",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    cfg: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    """SDXL Turbo kids-safe workflow — aggressive safety negatives baked in."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "persona_safe", "images": ["8", 0]},
        },
    }


# Map of workflow name → (function, description, min_vram_gb)
WORKFLOW_REGISTRY = {
    "sd15_basic": (sd15_basic, "SD1.5 basic — fast, low VRAM", 4),
    "sdxl_turbo": (sdxl_turbo, "SDXL Turbo — fast, good quality", 10),
    "sd15_nsfw": (sd15_nsfw, "SD1.5 NSFW — adult only, no safety negatives", 4),
    "sdxl_safe": (sdxl_safe, "SDXL kids-safe — aggressive safety negatives", 10),
}


def get_workflow(name: str):
    """Get a workflow template by name. Returns (func, description, min_vram) or None."""
    return WORKFLOW_REGISTRY.get(name)


def list_workflows() -> list:
    """List all available workflow templates with descriptions."""
    return [
        {"name": name, "description": desc, "min_vram_gb": vram}
        for name, (_, desc, vram) in WORKFLOW_REGISTRY.items()
    ]
