import layers.conv2d
import layers.conv2d_transpose
import layers.dense
import layers.flatten
import layers.maxpool
import layers.reshape
import layers.resize
import tensorflow as tf

LAYER_BUILD_FNS = {
    "conv2d": layers.conv2d.build,
    "conv2d_transpose": layers.conv2d_transpose.build,
    "dense": layers.dense.build,
    "flatten": layers.flatten.build,
    "maxpool": layers.maxpool.build,
    "reshape": layers.reshape.build,
    "resize": layers.resize.build
}

def build_layer(prev_layer, layer_info, is_training):
    layer_type = layer_info["type"]
    if layer_type in LAYER_BUILD_FNS:
        build_fn = LAYER_BUILD_FNS[layer_type]
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
