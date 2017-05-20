DATASET_NAME = "CELEBA"

if DATASET_NAME == "MNIST":
    IMAGE_DIMS = (28, 28, 1)
    COLOR_MODE = "L"
elif DATASET_NAME == "CELEBA":
    IMAGE_DIMS = (28, 28, 3)
    COLOR_MODE = "RGB"
else:
    raise Exception(f"Unknown dataset name: {DATASET_NAME}")
