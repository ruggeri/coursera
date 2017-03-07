import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from lib.runner import Runner

def run_image(image_path):
    image = mpimg.imread(image_path)
    runner = Runner(image.shape)
    runner.run(image)

    plt.figure()
    plt.imshow(image)
    plt.show()

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

if __name__ == "__main__":
    #run_image("tests/jpgs/solidWhiteCurve.jpg")
    run_movie("challenge.mp4")
