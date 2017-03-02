import cv2
import matplotlib.pyplot as plt
import numpy as np

DEBUG_MODE = False

LAST_MASKED_IMAGE = None
LAST_MASKED_EDGES = None
LAST_RAW_LINES = None
BAD_THETA_LEFT_LINES = None
BAD_THETA_RIGHT_LINES = None
FILTERED_LEFT_LINES = None
FILTERED_RIGHT_LINES = None
LAST_POSTPROCESSED_LINES = None
LAST_OVERLAID_IMAGE = None

def draw_lines(image, lines, color, thickness):
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(
            image, (x1, y1), (x2, y2), color, thickness
        )

    return image

def show_last_images():
    all_images = [
        (LAST_MASKED_IMAGE, "IMAGE"),
        (LAST_MASKED_EDGES, "IMAGE"),
        (LAST_RAW_LINES, (255, 0, 0)),
        (FILTERED_LEFT_LINES, (0, 255, 0)),
        (FILTERED_RIGHT_LINES, (0, 255, 0)),
        (LAST_POSTPROCESSED_LINES, (0, 0, 255)),
        (LAST_OVERLAID_IMAGE, "IMAGE")
    ]

    for (x, t) in all_images:
        plt.figure()
        if t == "IMAGE":
            if len(x.shape) == 2:
                plt.imshow(x, cmap="gray")
            else:
                plt.imshow(x)
        else:
            overlay = np.zeros(list(LAST_MASKED_IMAGE.shape) + [3], dtype=np.uint8)
            draw_lines(overlay, x, 255, 2)
            plt.imshow(overlay)

    plt.show()
