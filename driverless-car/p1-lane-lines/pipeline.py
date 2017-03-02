import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os

import lib.config as config
from lib.edges import preprocess_image, detect_edges
from lib.lines import detect_lines

def draw_lines(image, lines, color, thickness):
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(
            image, (x1, y1), (x2, y2), color, thickness
        )

    return image

CACHED_BUFFER = None
def draw_transparent_lines(image, lines, color, thickness):
    global CACHED_BUFFER
    if (CACHED_BUFFER is None) or CACHED_BUFFER.shape != image.shape:
        CACHED_BUFFER = np.zeros(image.shape, image.dtype)

    CACHED_BUFFER *= 0
    draw_lines(CACHED_BUFFER, lines, color, thickness)
    cv2.addWeighted(image, 1.0, CACHED_BUFFER, 1.0, 0, image)
    return image

def run_pipeline(image):
    # Preprocess, detect edges, detect lines.
    masked_image = preprocess_image(image)
    masked_edges = detect_edges(masked_image)
    detected_lines = detect_lines(masked_edges)

    # Produce final overlaid image and display.
    height, width = image.shape[0], image.shape[1]
    draw_lines(
        image, [[int(width / 2), 0, int(width / 2), height]], (255, 255, 255), 2
    )

    draw_transparent_lines(
        image, detected_lines, (255, 0, 0), 20
    )
    draw_lines(
        image, config.LAST_RAW_LINES, (0, 255, 0), 2
    )
    draw_lines(
        image, config.BAD_THETA_LEFT_LINES, (0, 255, 255), 2
    )
    draw_lines(
        image, config.BAD_THETA_RIGHT_LINES, (0, 255, 255), 2
    )
    draw_lines(
        image, config.FILTERED_LEFT_LINES, (0, 0, 255), 2
    )
    draw_lines(
        image, config.FILTERED_RIGHT_LINES, (0, 0, 255), 2
    )

    config.LAST_OVERLAID_IMAGE = image
    return image

def run_pipeline_all():
    # Close previously opened images to reduce memory leak.
    plt.close("all")

    for path in os.listdir("test_images/"):
        image_path = "test_images/{}".format(path)
        print(image_path)

        # Read Image
        image = mpimg.imread(image_path)
        overlaid_image = run_pipeline(image)

        # Show the image.
        plt.figure()
        plt.imshow(overlaid_image)
        plt.show()

def test_movie():
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    import traceback

    def process_image(image):
        result = run_pipeline(image)
        return result

    try:
        challenge_output = 'extra.mp4'
        clip2 = VideoFileClip('challenge.mp4')
        challenge_clip = clip2.fl_image(process_image)
        challenge_clip.write_videofile(challenge_output, audio=False)
    except Exception as e:
        print(e)
        traceback.print_exc()
        config.show_last_images()

if __name__ == "__main__":
    config.DEBUG_MODE = False
    test_movie()
    #run_pipeline_all()
