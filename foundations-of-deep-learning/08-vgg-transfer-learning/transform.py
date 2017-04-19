from collections import namedtuple
import config
import numpy as np
import os
import pickle
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import vgg

RunInfo = namedtuple("RunInfo", [
    "session",
    "rgb_layers",
    "fc6",
])

def load_image(filename):
    img = skimage.io.imread(filename) / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop to center
    short_edge = min(img.shape[0], img.shape[1])
    top_y = (img.shape[0] // 2 - short_edge // 2)
    left_x = (img.shape[1] // 2 - short_edge // 2)
    img = img[top_y:(top_y + short_edge), left_x:(left_x + short_edge)]

    # Now resize to proper dimensions.
    img = skimage.transform.resize(img, config.VGG_IMG_DIMS[0:2])
    return img

def build_vgg_model():
    # Expects images to be of format 244x244x3.
    rgb_layers = tf.placeholder(
        tf.float32, [None, *config.VGG_IMG_DIMS]
    )
    vgg_model = vgg.VGG16Model(config.VGG_PARAMS_FILENAME)
    with tf.name_scope("vgg_model"):
        vgg_model.build(rgb_layers)
        return (rgb_layers, vgg_model.fc6)

def flower_class_names():
    flower_data_dir = config.FLOWER_DATA_DIR
    subdirs = os.listdir(flower_data_dir)
    flower_class_names = [
        dirname
        for dirname in subdirs if os.path.isdir(flower_data_dir + dirname)
    ]
    return flower_class_names

def num_flowers(flower_class_name):
    flower_files = os.listdir(
        config.FLOWER_DATA_DIR + flower_class_name
    )
    num_flowers = len(flower_files)
    return num_flowers

def flower_batches(flower_class_name):
    flower_files = os.listdir(
        config.FLOWER_DATA_DIR + flower_class_name
    )
    num_flowers = len(flower_files)

    flower_start_idxs = range(
        0, num_flowers, config.TRANSFORM_BATCH_SIZE
    )
    for flower_start_idx in flower_start_idxs:
        flower_end_idx = min(
            flower_start_idx + config.TRANSFORM_BATCH_SIZE,
            num_flowers
        )

        yield flower_files[flower_start_idx:flower_end_idx]

def transform_flower_batch(
        run_info,
        flower_class_name,
        flower_files):
    ri = run_info

    batch = []
    for flower_file in flower_files:
        fname = os.path.join(
            config.FLOWER_DATA_DIR, flower_class_name, flower_file
        )
        img = load_image(fname)
        batch.append(img.reshape(config.VGG_IMG_DIMS))

    imgs = np.array(batch)
    codes = ri.session.run(ri.fc6, feed_dict = {
        ri.rgb_layers: imgs
    })

    return codes

def transform_flower_class(
        run_info,
        flower_class_name):
    codes = []
    batches = flower_batches(flower_class_name)
    num_batches = (
        num_flowers(flower_class_name) // config.TRANSFORM_BATCH_SIZE
    )
    for idx, flower_batch_files in enumerate(batches):
        print(f"Beginning batch {idx}/{num_batches} of "
              f"{flower_class_name}!")
        batch_codes = transform_flower_batch(
            run_info, flower_class_name, flower_batch_files
        )
        codes.append(batch_codes)

    codes = np.concatenate(codes)
    return codes

def transform_flower_classes(run_info):
    label_idx_to_flower_class_name = {}
    codes = []
    labels = []
    for flower_class_name in flower_class_names():
        print(f"Beginning transformation of {flower_class_name} files!")
        label_idx = len(label_idx_to_flower_class_name)
        label_idx_to_flower_class_name[flower_class_name] = label_idx

        class_codes = transform_flower_class(
            run_info, flower_class_name
        )
        class_labels = np.zeros(class_codes.shape[0], dtype=np.int32)
        class_labels.fill(label_idx)

        codes.append(class_codes)
        labels.append(class_labels)

    codes = np.concatenate(codes)
    labels = np.concatenate(labels)

    return (codes, labels, label_idx_to_flower_class_name)

def transform(session):
    rgb_layers, fc6 = build_vgg_model()
    run_info = RunInfo(
        session = session,
        rgb_layers = rgb_layers,
        fc6 = fc6,
    )

    codes, labels, label_idx_to_flower_class_name = (
        transform_flower_classes(run_info)
    )

    np.save(config.CODES_FILENAME, codes)
    np.save(config.LABELS_FILENAME, labels)
    with open(config.LABELS_MAP_FILENAME, "wb") as f:
        pickle.dump(label_idx_to_flower_class_name, f)

with tf.Session() as session:
    transform(session)
