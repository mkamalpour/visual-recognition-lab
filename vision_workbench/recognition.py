"""Inspectable cosine classification; fitting never needs query embeddings."""

import numpy as np


def normalize_rows(values):
    if np.iscomplexobj(values):
        raise ValueError("Embeddings must be real; complex components cannot be discarded.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Expected a nonempty two-dimensional embedding array.")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings must be finite.")
    scales = np.max(np.abs(array), axis=1, keepdims=True)
    if (scales == 0).any():
        raise ValueError("Zero-length embeddings cannot define cosine similarity.")
    scaled = array / scales
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def build_classifier(
    text_embeddings,
    reference_embeddings=None,
    augmented_embeddings=None,
    alpha=0.7,
    keep=None,
):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one.")
    text_embeddings = list(text_embeddings)
    if not text_embeddings:
        raise ValueError("Supply at least one class of text embeddings.")
    text = normalize_rows(
        np.stack([normalize_rows(prompts).mean(axis=0) for prompts in text_embeddings])
    )
    n, dim = text.shape
    if keep is None:
        keep = n
    if (
        isinstance(keep, bool)
        or not isinstance(keep, (int, np.integer))
        or not 1 <= keep <= n
    ):
        raise ValueError("keep must be an integer between one and the class count.")
    if reference_embeddings is None:
        if augmented_embeddings is not None:
            raise ValueError("Augmentations require reference embeddings.")
        indices = np.arange(n)[:keep]
        return dict(
            indices=indices,
            weights=text[indices].copy(),
            text_weights=text[indices].copy(),
            visual_weights=text[indices].copy(),
            counts=np.zeros(keep, dtype=int),
            assignments=np.array([], dtype=int),
        )
    refs = normalize_rows(reference_embeddings)
    if refs.shape[1] != dim:
        raise ValueError("Text and reference dimensions differ.")
    mean = refs.mean(axis=0)
    indices = np.sort(np.argsort(-(text @ mean), kind="stable")[:keep])
    selected = text[indices]
    assignments = np.argmax(refs @ selected.T, axis=1)
    if augmented_embeddings is None:
        augmented = refs[:, None, :]
    else:
        if np.iscomplexobj(augmented_embeddings):
            raise ValueError("Augmented embeddings must be real.")
        augmented = np.asarray(augmented_embeddings, dtype=float)
        if (
            augmented.ndim != 3
            or augmented.shape[0] != len(refs)
            or augmented.shape[2] != dim
            or not augmented.shape[1]
        ):
            raise ValueError(
                "Augmentations must have shape (references, views, dimensions)."
            )
        augmented = normalize_rows(augmented.reshape(-1, dim)).reshape(augmented.shape)
    counts = np.bincount(assignments, minlength=keep)
    visual = selected.copy()
    for index in range(keep):
        if counts[index]:
            visual[index] = normalize_rows(
                augmented[assignments == index]
                .reshape(-1, dim)
                .mean(axis=0, keepdims=True)
            )[0]
    weights = normalize_rows(alpha * selected + (1 - alpha) * visual)
    return dict(
        indices=indices,
        weights=weights,
        text_weights=selected,
        visual_weights=visual,
        counts=counts,
        assignments=indices[assignments],
    )


def rank(query, classifier, labels, top_k=3, temperature=0.07):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    query = np.asarray(query)
    if query.ndim == 1:
        query = query[None, :]
    query = normalize_rows(query)
    weights = normalize_rows(classifier["weights"])
    if len(query) != 1 or query.shape[1] != weights.shape[1]:
        raise ValueError("Supply one query with the classifier dimension.")
    indices = np.asarray(classifier["indices"])
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(indices) != len(weights) or len(np.unique(indices)) != len(indices)
            or any(i < 0 or i >= len(labels) for i in indices)):
        raise ValueError("Classifier indices do not match labels.")
    similarity = (query @ weights.T)[0]
    logits = (similarity - similarity.max()) / temperature
    scores = np.exp(logits)
    scores /= scores.sum()
    order = np.argsort(-similarity, kind="stable")[:top_k]
    return [
        dict(
            label=str(labels[indices[i]]),
            score=float(scores[i]),
            similarity=float(similarity[i]),
        )
        for i in order
    ]


def evaluate_predictions(expected, predicted, labels):
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Labels must be nonempty and unique.")
    expected, predicted = list(expected), list(predicted)
    if not expected or len(expected) != len(predicted):
        raise ValueError(
            "Expected and predicted labels must have matching nonzero lengths."
        )
    lookup = {label: i for i, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, guess in zip(expected, predicted):
        if truth not in lookup or guess not in lookup:
            raise ValueError("Unknown evaluation label.")
        confusion[lookup[truth], lookup[guess]] += 1
    return dict(
        accuracy=float(np.trace(confusion) / len(expected)),
        confusion=confusion.tolist(),
        labels=labels,
        count=len(expected),
    )
