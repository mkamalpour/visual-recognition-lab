"""Deterministic reference preparation, separate from model execution."""

import hashlib
import io
import random
import warnings

from PIL import Image, ImageOps


def decode_image(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")


def image_digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).digest()


def reference_images(query, references):
    query_hash = image_digest(decode_image(query))
    seen = set()
    result = []
    for data in references:
        image = decode_image(data)
        digest = image_digest(image)
        if digest == query_hash:
            raise ValueError(
                "A reference duplicates the query pixels. Remove it to prevent data leakage."
            )
        if digest not in seen:
            result.append(image)
            seen.add(digest)
    return result


def image_variants(image, count):
    if not 1 <= count <= 8:
        raise ValueError("Use between 1 and 8 reference views.")
    result = [image.copy()]
    rng = random.Random(42)
    for index in range(count - 1):
        width, height = image.size
        crop_width = max(1, int(width * rng.uniform(0.8, 0.95)))
        crop_height = max(1, int(height * rng.uniform(0.8, 0.95)))
        left = rng.randrange(width - crop_width + 1)
        top = rng.randrange(height - crop_height + 1)
        crop = image.crop((left, top, left + crop_width, top + crop_height)).resize(
            image.size
        )
        result.append(ImageOps.mirror(crop) if index % 2 == 0 else crop)
    return result


def validate_descriptions(names, descriptions):
    """Validate all supplied prompts before any model download or execution."""
    if not isinstance(descriptions, dict):
        raise ValueError(
            "Descriptions JSON must map candidate names to lists of text prompts."
        )
    if len(descriptions) > 1000:
        raise ValueError("Use at most 1000 description entries.")
    for name, prompts in descriptions.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(prompts, list)
            or not 1 <= len(prompts) <= 100
            or any(
                not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
            )
        ):
            raise ValueError(
                "Each candidate description must be a nonempty list of at most 100 strings."
            )
    return [
        descriptions.get(name, [f"a photo of a {name}", f"an image of a {name}"])
        for name in names
    ]
