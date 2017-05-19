import config
import layers.conv2d
import layers.dense
import layers.flatten
import layers.maxpool
import layers.reshape
import layers.resize
import tensorflow as tf

def build_layer(prev_layer, layer_info, is_training):
    layer_type = layer_info["type"]
    build_fn = None
    if layer_type == "conv2d":
        build_fn = layers.conv2d.build
    elif layer_type == "dense":
        build_fn = layers.dense.build
    elif layer_type == "flatten":
        build_fn = layers.flatten.build
    elif layer_type == "maxpool":
        build_fn = layers.maxpool.build
    elif layer_type == "reshape":
        build_fn = layers.reshape.build
    elif layer_type == "resize":
        build_fn = layers.resize.build
    else:
        raise Exception(f"unknown layer type: {layer_type}")

    return build_fn(prev_layer, layer_info, is_training)

def build_layers(initial_layer, layer_infos, is_training):
    prev_layer = initial_layer
    for layer_info in layer_infos:
        prev_layer = build_layer(
            prev_layer,
            layer_info,
            is_training = is_training
        )

    return prev_layer
