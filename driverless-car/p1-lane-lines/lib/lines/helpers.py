import math

def line_length(line):
    dx = line[0] - line[2]
    dy = line[1] - line[3]

    return math.sqrt(dx*dx + dy*dy)

def line_theta(line):
    dx = line[2] - line[0]
    dy = line[3] - line[1]

    return math.atan2(abs(dy), abs(dx))

def weighted_median(values_and_weights):
    if len(values_and_weights) == 0:
        raise Exception("Why no values and weights?")

    values_and_weights = sorted(
        values_and_weights, key=lambda pair: pair[0]
    )

    total_weight = 0
    for _, weight in values_and_weights:
        total_weight += weight

    target_weight = total_weight / 2
    seen_weight = 0
    for value, weight in values_and_weights:
        next_weight = seen_weight + weight
        should_stop = (
            (seen_weight <= target_weight)
            and (target_weight <= next_weight)
        )
        if should_stop:
            return value
        seen_weight = next_weight

    raise Exception("Never found median?")
