import numpy as np
import tensorflow as tf

VGG_BGR_MEANS = [103.939, 116.779, 123.68]

class VGG16Model:
    def __init__(self, npy_file_path):
        # A dictionary was saved as an ndarray scalar of type
        # object. That way they could save it using the npy format.
        self.parameters = np.load(npy_file_path, encoding="latin1").item()

    def build(self, rgb_layers):
        # Assumes your input is 0-1.0. First we put in range 0-255.
        self.rgb_layers = rgb_layers * 255.0

        # Splits out input into layers so we can mean normalize and
        # put into BGR format.
        red_layer, green_layer, blue_layer = (
            tf.split(self.rgb_layers, axis=3)
        )

        bgr_layers_list = [
            blue_layer - VGG_BGR_MEANS[0],
            green_layer - VGG_BGR_MEANS[1],
            red_layer - VGG_BGR_MEANS[2]
        ]
        self.bgr_layers = tf.concat(
            bgr_layers_list,
            axis = 3
        )

        # Extracts 64 feature maps.
        self.conv1_1 = self.conv_layer("conv1_1", self.bgr_layers)
        self.conv1_2 = self.conv_layer("conv1_2", self.conv1_1)
        self.pool1 = self.max_pool_layer("pool1", self.conv1_2)

        # Extracts 128 feature maps.
        self.conv2_1 = self.conv_layer("conv2_1", self.pool1)
        self.conv2_2 = self.conv_layer("conv2_2", self.conv2_1)
        self.pool2 = self.max_pool_layer("pool2", self.conv2_2)

        # Extracts 256 feature maps.
        self.conv3_1 = self.conv_layer("conv3_1", self.pool2)
        self.conv3_2 = self.conv_layer("conv3_2", self.conv3_1)
        self.conv3_3 = self.conv_layer("conv3_3", self.conv3_2)
        self.pool3 = self.max_pool_layer("pool3", self.conv3_3)

        # Extracts 512 feature maps.
        self.conv4_1 = self.conv_layer("conv4_1", self.pool3)
        self.conv4_2 = self.conv_layer("conv4_2", self.conv4_1)
        self.conv4_3 = self.conv_layer("conv4_3", self.conv4_2)
        self.pool4 = self.max_pool_layer("pool4", self.conv4_3)

        # Extracts 512 feature maps.
        self.conv5_1 = self.conv_layer("conv5_1", self.pool4)
        self.conv5_2 = self.conv_layer("conv5_2", self.conv5_1)
        self.conv5_3 = self.conv_layer("conv5_2", self.conv5_2)
        self.pool5 = self.max_pool_layer("pool5", self.conv5_3)

        # Two fully-connected layers of 4096 units each. RELU is
        # applied.
        self.fc6 = self.fully_connected_layer("fc6", self.pool5)
        self.fc6 = tf.nn.relu(self.fc6)
        self.fc7 = self.fully_connected_layer("fc7", self.fc6)
        self.fc7 = tf.nn.relu(self.fc7)

        # Final layer of 1k units. fc8 is the logits.
        self.fc8 = self.fully_connected_layer("fc8", self.fc7)
        self.probabilities = tf.nn.softmax(fc8)

    def conv_layer(self, layer_name, layer_input):
        # All convolutions are 3x3. Step size of one in each
        # direction. Same padding.

        filter_weights = self.get_weights(layer_name)
        filter_biases = self.get_biases(layer_name)
        conv_layer = tf.nn.conv2d(
            layer_input,
            filter_weights,
            strides = [1, 1, 1, 1],
            padding = "SAME"
        )

        conv_layer = tf.add_bias(conv_layer, filter_biases)
        conv_layer = tf.nn.relu(conv_layer)

        return conv_layer

    def fc_layer(self, layer_name, layer_input):
        weights = self.get_weights(layer_name)
        biases = self.get_biases(layer_name)

        # may need to flatten
        layer_input_dims = layer_input.get_shape().as_list()
        if len(layer_input_dims) > 2:
            num_features = 1
            for dim in layer_input_dims[1:]:
                num_features *= dim
            layer_input = tf.reshape(layer_input, (-1, num_features))

        return (tf.matmul(layer_input, weights) + biases)

    def max_pool_layer(self, layer_name, layer_input):
        # All pooling is 2x2.
        return tf.nn.max_pool(
            layer_input,
            ksize = [1, 2, 2, 1],
            strides = [1, 2, 2, 1],
            padding = "SAME"
        )

    def get_weights(name):
        return tf.constant(self.parameters[name][0])

    def get_biases(name):
        return tf.constant(self.parameters[name][1])
