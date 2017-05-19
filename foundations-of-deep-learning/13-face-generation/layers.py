import config
import tensorflow as tf

def build_layer(prev_layer, layer_info, is_training):
    build_fn = None
    if layer_info["type"] == "conv2d":
        build_fn = build_conv2d
    elif layer_info["type"] == "maxpool":
        build_fn = build_maxpool
    elif layer_info["type"] == "reshape":
        build_fn = build_reshape
    elif layer_info["type"] == "resize":
        build_fn = build_resize
    else:
        raise Exception("unknown layer type")

    return build_fn(prev_layer, layer_info, is_training)

def build_layers(initial_layer, layer_infos, is_training):
    prev_layer = initial_layer
    for layer_info in layer_infos:
        prev_layer = helper.build_layer(
            prev_layer,
            layer_info,
            is_training = is_training
        )

    return prev_layer
