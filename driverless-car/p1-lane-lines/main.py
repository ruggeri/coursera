import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import os

from lib.runner import Runner

def run_image(image_path):
    print(image_path)

    image = mpimg.imread(image_path)
    runner = Runner(image.shape)
    runner.run(image)

    plt.figure()
    plt.imshow(image)
    plt.show()

def run_images():
    for fname in os.listdir("tests/jpgs/"):
        image_path = "tests/jpgs/{}".format(fname)
        run_image(image_path)

class RunnerWrapper:
    def __init__(self):
        self.runner = None

    def run(self, image):
        if self.runner is None:
            runner = Runner(image.shape)
        return runner.run(image)

def run_movie(name):
    input_path = "tests/mp4s/{}".format(name)
    output_path = "output/{}".format(name)

    runner = RunnerWrapper()

    clip2 = VideoFileClip(input_path)
    challenge_clip = clip2.fl_image(runner.run)
    challenge_clip.write_videofile(output_path, audio = False)

def run_movies():
    for fname in os.listdir("tests/mp4s/"):
        run_movie(fname)

if __name__ == "__main__":
    #run_images()
    #run_movies()
    #run_movie("solidWhiteRight.mp4")
    #run_movie("solidYellowLeft.mp4")
    #run_movie("challenge.mp4")
