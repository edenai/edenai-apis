from typing import List

import numpy as np

SCORE_MULTIPLIER = 100.0


def cosine_similarity(embedding1: List[float], embedding2: List[float]):
    """
    Computes the cosine similarity between two vectors.
    """
    vect_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_similarity = vect_product / (norm1 * norm2)
    return cosine_similarity * SCORE_MULTIPLIER


def manhattan_similarity(embedding1: List[float], embedding2: List[float]):
    """
    Computes the manhattan similarity between two vectors.
    """
    return SCORE_MULTIPLIER - sum(
        abs(val1 - val2) for val1, val2 in zip(embedding1, embedding2)
    )


def squared_euclidean_similarity(embedding1: List[float], embedding2: List[float]):
    """
    Computes the euclidean similarity between two vectors.
    """
    # numpy arrays
    point1 = np.array(embedding1)
    point2 = np.array(embedding2)

    # calculating Euclidean distance
    dist = np.linalg.norm(point1 - point2)
    return (1 - dist) * SCORE_MULTIPLIER


METRICS = {
    "cosine": cosine_similarity,
    "manhattan": manhattan_similarity,
    "euclidean": squared_euclidean_similarity,
}
