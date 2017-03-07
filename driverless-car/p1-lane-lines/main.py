import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from lib.runner import Runner

def run_image(image_path):
    image = mpimg.imread(image_path)
    runner = Runner(image.shape)
    runner.run(image)

    plt.figure()
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    run_image("tests/jpgs/solidWhiteCurve.jpg")
