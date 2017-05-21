import config

if config.IMAGE_DIMS[2] == 1:
    COLOR_MODE = "L"
elif config.IMAGE_DIMS[2] == 3:
    COLOR_MODE = "RGB"
else:
    raise Exception("Unrecognized color depth!")

NUM_SAMPLES_PER_SAMPLING = 9
