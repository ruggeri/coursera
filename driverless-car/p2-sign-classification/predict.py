import csv
import dataset as dataset_mod
import matplotlib.pyplot as plt
import network as network_mod
import numpy as np
import tensorflow as tf
import train

NUM_EXAMPLES = 10
TOP_K = 5

def predict(session, network, examples):
    predictions = session.run(
        tf.nn.softmax(network.logits), feed_dict = {
            network.x: examples,
            network.keep_prob: 1.0,
            network.training: False,
        }
    )

    return predictions

def load_sign_names():
    sign_names_map = {}
    with open("data/signnames.csv", "r") as f:
        for row in csv.DictReader(f, delimiter = ","):
            sign_names_map[int(row["ClassId"])] = row["SignName"]
    return sign_names_map

# I load a number of test images that have never been seen by the
# learner.
def select_examples(dataset):
    example_idxs = np.random.choice(
        dataset.X_test.shape[0], size = NUM_EXAMPLES, replace = False
    )
    examples_x = dataset.X_test[example_idxs, :, :, :]
    examples_y = dataset.y_test[example_idxs]

    return examples_x, examples_y

def display_results(examples_x, examples_y, predictions):
    sign_names_map = load_sign_names()

    num_correct_predictions = 0
    num_topK_correct_predictions = 0
    for example_idx, (x, y) in enumerate(zip(examples_x, examples_y)):
        # imshow doesn't want a 3d image if grayscale.
        plt.imshow(x.squeeze(2), cmap = "gray")
        plt.show()

        print(f"True class: {sign_names_map[y]}")

        topK_classes = np.argsort(-predictions[example_idx])[:TOP_K]
        for prediction_idx in range(TOP_K):
            predicted_class_idx = topK_classes[prediction_idx]
            predicted_class_name = sign_names_map[predicted_class_idx]
            predicted_prob = predictions[
                example_idx, predicted_class_idx
            ]
            print(f"Guess #{prediction_idx} | {predicted_class_name} | "
                  f"{100*predicted_prob:3.1f}%f")

            if y == predicted_class_idx:
                num_topK_correct_predictions += 1

        if y == topK_classes[0]:
            num_correct_predictions += 1

    accuracy = num_correct_predictions / NUM_EXAMPLES
    topK_accuracy = num_topK_correct_predictions / NUM_EXAMPLES
    print(f"Test accuracy: {100*accuracy:3.1f}")
    print(f"Top {TOP_K} accuracy: {100*topK_accuracy:3.1f}")

def main():
    with tf.Session() as session:
        dataset = dataset_mod.load()
        network = network_mod.build_network(
            dataset.image_shape,
            dataset.num_classes
        )

        saver = tf.train.Saver()
        saver.restore(session, "models/model.ckpt")

        examples_x, examples_y = select_examples(dataset)
        predictions = predict(session, network, examples_x)
        display_results(examples_x, examples_y, predictions)

if __name__ == "__main__":
    main()
