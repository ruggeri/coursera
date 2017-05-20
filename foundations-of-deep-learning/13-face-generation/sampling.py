import config.sampling
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def sample_generator_output(session, network):
    num_z_dimensions = network.fake_z.get_shape()[1]
    fake_z = np.random.uniform(
        low = -1.0,
        high = +1.0,
        size = (
            config.sampling.NUM_SAMPLES_PER_SAMPLING,
            num_z_dimensions
        )
    )

    samples = session.run(
        network.inference_fake_x,
        feed_dict = {
            network.fake_z: fake_z
        }
    )

    transformed_samples = np.zeros_like(
        samples,
        dtype = np.uint8
    )
    for idx, raw_sample in enumerate(samples):
        # Convert to 8bit format.
        transformed_sample = ((raw_sample + 1) / 2) * 255
        transformed_sample = transformed_sample.astype(np.uint8)
        transformed_samples[idx] = transformed_sample

    if config.sampling.IMAGE_DIMS[2] == 1:
        # imshow only likes to show 3d images if the third
        # dimension is 3 or 4. If it's 1, it just wants a 2d
        # image.
        transformed_samples = np.squeeze(transformed_samples, 3)

    return transformed_samples

def save_image(title, image):
    if plt.isinteractive():
        plt.figure()
    # This is just to handle black/white images. cmap means
    # contrast map I think.
    if config.sampling.COLOR_MODE == "RGB":
        cmap = None
    elif config.sampling.COLOR_MODE == "L":
        cmap = "gray"
    plt.imshow(image, cmap = cmap)
    plt.title(title)
    plt.savefig(f"samples/{title}.png")
    if plt.isinteractive():
        plt.show()
    else:
        plt.close()

def save_samples(epoch_idx, batch_idx, samples):
    for sample_idx, sample in enumerate(samples):
        fname = (
            f"sample_e{epoch_idx:02d}_b{batch_idx:04d}_{sample_idx:02d}"
        )
        save_image(fname, sample)

def build_samples_grid(samples):
    # creating a square grid of images. may have to throw some away if
    # num samples is not square.
    image_grid_dimension = int(np.sqrt(samples.shape[0]))
    num_samples_in_grid = image_grid_dimension * image_grid_dimension

    # Put samples in a square arrangement. I am careful not to use
    # config.IMAGE_SHAPE because imshow is not going to like having a
    # dimension of size 1 for black/white.
    image_shape = samples.shape[1:]
    samples_in_square = np.reshape(
        samples[:num_samples_in_grid],
        (image_grid_dimension, image_grid_dimension, *image_shape)
    )

    # Combine samples to grid image
    samples_grid = Image.new(
        config.sampling.COLOR_MODE,
        (samples.shape[1] * image_grid_dimension,
         samples.shape[2] * image_grid_dimension)
    )
    for col_idx in range(image_grid_dimension):
        for row_idx in range(image_grid_dimension):
            image = Image.fromarray(
                samples_in_square[row_idx, col_idx],
                config.sampling.COLOR_MODE
            )
            # Weird. It looks like PIL uses x,y coordinates. Worst
            # case scenario is that this prints the samples into a
            # transposed grid. That's fine too.
            samples_grid.paste(
                image,
                (col_idx * samples.shape[1],
                 row_idx * samples.shape[2])
            )

    return samples_grid

def save_samples_grid(epoch_idx, batch_idx, samples_grid):
    fname = (
        f"sample_e{epoch_idx:02d}_b{batch_idx:04d}_grid"
    )
    save_image(fname, samples_grid)

def run(epoch_idx, batch_idx, session, network):
    samples = sample_generator_output(session, network)
    save_samples(epoch_idx, batch_idx, samples)
    samples_grid = build_samples_grid(samples)
    save_samples_grid(epoch_idx, batch_idx, samples_grid)
