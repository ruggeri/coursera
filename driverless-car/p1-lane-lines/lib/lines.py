import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

import lib.config as config

# For smoothing across frames
PREV_LEFT_LINES = []
PREV_RIGHT_LINES = []
NUM_LINES_HISTORY = 5

# Hough Constants
RHO_ACCURACY = 1
THETA_ACCURACY = (2 * np.pi) / 180
HOUGH_THRESHOLD = 10
HOUGH_MIN_LINE_LENGTH = 10
HOUGH_MAX_LINE_GAP = 2

# How far up to extend line?
HORIZON_RATIO = 0.35

# Detect lines in the image, averaging and extending them.
def detect_lines(masked_edges):
    height, width = masked_edges.shape

    # Detect lines.
    lines = cv2.HoughLinesP(
        masked_edges,
        rho = RHO_ACCURACY,
        theta = THETA_ACCURACY,
        threshold = HOUGH_THRESHOLD,
        minLineLength = HOUGH_MIN_LINE_LENGTH,
        maxLineGap = HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        raise Exception("No lines detected!")

    # Unpack the lines from unneeded nesting.
    lines = [line[0] for line in lines]

    # Display intermediate result of line detection before
    # postprocessing.
    config.LAST_RAW_LINES = lines
    if config.DEBUG_MODE:
        overlay = draw_lines(height, width, lines, thickness = 2)
        plt.figure()
        plt.imshow(overlay)

    # Filter and average a left and right line.
    lines = postprocess_lines(lines, width, height, HORIZON_RATIO)

    # Display intermediate result after averaging and extension.
    config.LAST_POSTPROCESSED_LINES = lines
    if config.DEBUG_MODE:
        overlay = draw_lines(height, width, lines)
        plt.figure()
        plt.imshow(overlay)

    # TODO: Very wasteful to draw here!
    return lines

def postprocess_lines(lines, width, height, horizon_ratio):
    # First average a left and right line.
    left_lines, right_lines = split_lines_by_half(lines, width)

    if len(left_lines) == 0:
        raise Exception("No left lines detected!")
    if len(right_lines) == 0:
        raise Exception("No right lines detected!")

    left_lines = filter_extreme_lines(left_lines, height, "LEFT")
    right_lines = filter_extreme_lines(right_lines, height, "RIGHT")

    # Show lines after filtering.
    config.FILTERED_LEFT_LINES = left_lines
    config.FILTERED_RIGHT_LINES = right_lines
    if config.DEBUG_MODE:
        left_overlay = draw_lines(height, width, left_lines, thickness = 2)
        plt.figure()
        plt.imshow(left_overlay)
        right_overlay = draw_lines(height, width, right_lines, thickness = 2)
        plt.figure()
        plt.imshow(right_overlay)

    horizon_height = int(height * (1 - horizon_ratio))
    left_line = average_line(left_lines, horizon_height, height, "LEFT")
    right_line = average_line(right_lines, horizon_height, height, "RIGHT")

    left_line = smooth(left_line, PREV_LEFT_LINES)
    right_line = smooth(right_line, PREV_RIGHT_LINES)

    # Then extend left and right lines to the viewer.
    horizon_height = int(height * (1 - horizon_ratio))
    left_line = extend_line(left_line, horizon_height, height)
    right_line = extend_line(right_line, horizon_height, height)

    return (left_line, right_line)

DECAY_CONSTANT = 0.5
def smooth(line, lines):
    if len(lines) == 0:
        new_line = np.array(line)
        lines.append(new_line)
        return np.array(new_line)

    new_line = (np.array(line) + (lines[-1] * DECAY_CONSTANT)) / (1 + DECAY_CONSTANT)
    lines.append(new_line)
    if len(lines) > NUM_LINES_HISTORY:
        lines.pop(0)

    return new_line

SAFETY_MARGIN = 0.025
# Splits lines to lines on the "left half" and "right half" of the
# view.
def split_lines_by_half(lines, width):
    left_lines = []
    right_lines = []

    min_x = (width / 2) - (width * SAFETY_MARGIN)
    max_x = (width / 2) + (width * SAFETY_MARGIN)
    for line in lines:
        if line[0] < min_x and (line[2] < min_x):
            left_lines.append(line)
        elif (line[0] > max_x) and (line[2] > max_x):
            right_lines.append(line)

    return (left_lines, right_lines)

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

    print(values_and_weights)
    raise Exception("Never found median?")

def line_length(line):
    dx = line[0] - line[2]
    dy = line[1] - line[3]
    return math.sqrt(dx*dx + dy*dy)

def line_rho(line):
    slope = (line[3] - line[1]) / (line[2] - line[0])
    intercept = (-line[0] * slope) + line[1]
    return inte

def line_theta(line):
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    return math.atan2(abs(dy), abs(dx))

def line_weight(line, image_height):
    return line_length(line)

THETA_DIFF_THRESHOLD = 10 * ((2 * np.pi) / 360)
THETA_MIN = 15 * ((2 * np.pi) / 360)
TOO_DIFFERENT_PENALTY = 0.25
def filter_extreme_lines(lines, image_height, side):
    if len(lines) == 0:
        raise Exception("Should not be trying to filter zero lines...")

    bad_theta_lines = []
    prev_line_theta = None
    if side == "LEFT":
        config.BAD_THETA_LEFT_LINES = bad_theta_lines
        if len(PREV_LEFT_LINES) > 0:
            prev_line_theta = line_theta(PREV_LEFT_LINES[-1])
    else:
        config.BAD_THETA_RIGHT_LINES = bad_theta_lines
        if len(PREV_RIGHT_LINES) > 0:
            prev_line_theta = line_theta(PREV_RIGHT_LINES[-1])

    lines2 = []
    values_and_weights = []
    for line in lines:
        line = np.array(line)
        theta = line_theta(line)

        if (abs(theta) < THETA_MIN):
            bad_theta_lines.append(line)
            continue

        lines2.append(line)
        weight = line_weight(line, image_height)

        should_penalize = (
            (prev_line_theta is not None) and
            (abs(prev_line_theta - theta) > THETA_DIFF_THRESHOLD)
        )
        if should_penalize:
            weight *= TOO_DIFFERENT_PENALTY

        values_and_weights.append((theta, weight))
    lines = lines2

    median_theta = weighted_median(values_and_weights)

    filtered_lines = []
    for idx, line in enumerate(lines):
        theta = values_and_weights[idx][0]
        if abs(theta - median_theta) < THETA_DIFF_THRESHOLD:
            filtered_lines.append(line)

    return filtered_lines

TOO_DIFFERENT_X = 30
def too_different_from_prev(line, min_height, max_height, side):
    if side == "LEFT":
        if len(PREV_LEFT_LINES) == 0:
            return False
        prev_line = PREV_LEFT_LINES[-1]
    else:
        if len(PREV_RIGHT_LINES) == 0:
            return False
        prev_line = PREV_RIGHT_LINES[-1]

    line = extend_line(line, min_height, max_height)
    if line[1] > line[3]:
        raise Exception("Assumed higher y coordinate later?")

    delta_x = abs(prev_line[2] - line[2])
    return delta_x > TOO_DIFFERENT_X

OLD_LINE_FIT_MIX = 0.25
def average_line(lines, min_height, max_height, side):
    x = []
    y = []
    w = []

    total_weight = 0
    for line in lines:
        weight = line_length(line)
        if too_different_from_prev(line, min_height, max_height, side):
            weight *= TOO_DIFFERENT_PENALTY
        total_weight += weight

        x0, y0, x1, y1 = line
        x.append(x0)
        y.append(y0)
        w.append(weight)
        x.append(x1)
        y.append(y1)
        w.append(weight)

    prev_line = None
    if side == "LEFT" and len(PREV_LEFT_LINES) > 0:
        prev_line = PREV_LEFT_LINES[-1]
    elif side == "RIGHT" and len(PREV_RIGHT_LINES) > 0:
        prev_line = PREV_RIGHT_LINES[-1]

    if prev_line is not None:
        x0, y0, x1, y1 = extend_line(prev_line, min_height, max_height)
        weight = line_length(prev_line) / 8

        x.append(x0)
        y.append(y0)
        w.append(weight)
        x.append(x1)
        y.append(y1)
        w.append(weight)

    # This is tricky and predicts x from y!
    p = np.polyfit(y, x, 1, w = w)
    min_x = p[0] * min_height + p[1]
    max_x = p[0] * max_height + p[1]
    return [min_x, min_height, max_x, max_height]

# Extends a detected line all the way from `y=min_height` to
# `y=max_height`.
def extend_line(line, min_height, max_height):
    x1, y1 = line[:2]
    x2, y2 = line[2:]

    # Make (x1, y1) the more distant point, and (x2, y2) the
    # closer one. I really should be using Hesse normal form.
    if y1 > y2:
        (x2, y2), (x1, y1) = (x1, y1), (x2, y2)

    # This logic is technically incorrect for lines parallel to
    # the y-axis.
    if x1 == x2:
        raise Exception("Endpoints have same x value??")
    if y1 == y2:
        raise Exception("Endpoints have same y value??")

    m = (y2 - y1) / (x2 - x1)

    # Calculate new endpoints.
    y0 = min_height
    x0 = int(x1 + (y0 - y1) / m)
    y3 = max_height
    x3 = int(x2 + ((y3 - y2) / m))

    new_line = [x0, y0, x3, y3]
    return new_line

LINE_THICKNESS = 20
def draw_lines(height, width, lines, thickness = LINE_THICKNESS):
    overlay = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(
            overlay, (x1, y1), (x2, y2), 255, thickness
        )

    return overlay
