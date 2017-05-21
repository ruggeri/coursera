DATASET_NAME = "CELEBA"
NETWORK_NAME = "B"

if DATASET_NAME == "MNIST":
    IMAGE_DIMS = (28, 28, 1)
elif DATASET_NAME == "CELEBA":
    IMAGE_DIMS = (28, 28, 3)
else:
    raise Exception(f"Unknown dataset name: {DATASET_NAME}")
