import math


def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot / (norm1 * norm2)


def compute_hallushift_score(current_hidden_state, previous_hidden_state):
    """
    Prototype HalluShift score.
    Measures token-to-token internal shift using cosine distance.
    Output range: 0 to 1
    """
    if previous_hidden_state is None:
        return 0.0

    sim = cosine_similarity(current_hidden_state, previous_hidden_state)
    distance = 1.0 - sim

    if distance < 0.0:
        distance = 0.0
    if distance > 1.0:
        distance = 1.0

    return round(distance, 4)