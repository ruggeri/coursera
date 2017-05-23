import dataset as dataset_mod
import tensorflow as tf
import train as train_mod

if __name__ == "__main__":
    with tf.Session() as session:
        dataset = dataset_mod.load()
        train_mod.train(session, dataset)
