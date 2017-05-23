import config
import dataset as dataset_mod
import pickle
import tensorflow as tf
import train as train_mod

def load_dataset():
    training_file = f"{config.DATA_DIR}/train_augmented.p"
    validation_file= f"{config.DATA_DIR}/valid.p"
    testing_file = f"{config.DATA_DIR}/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return dataset_mod.build_dataset(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test
    )

with tf.Session() as session:
    dataset = load_dataset()
    train_mod.train(session, dataset)
