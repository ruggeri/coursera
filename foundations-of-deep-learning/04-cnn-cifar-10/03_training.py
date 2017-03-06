import tensorflow as tf
import time

from network import build_network

def make_batches(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

# We've stored our dataset on disk in five "segments." This method
# loads one segment and splits it up into batches.
NUM_SEGMENTS = 1
def load_segment_in_batches(segment_id, batch_size):
    filename = f"pickle-files/preprocessed_segment_{segment_id}.p"
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return list(make_batches(features, labels, batch_size))

# CONSTANTS
NUM_EPOCHS = 64
BATCH_SIZE = 256
KEEP_PROBABILITY = 0.50

class Trainer:
    def __init__(
            self,
            session,
            batch_size = BATCH_SIZE,
            keep_probability = KEEP_PROBABILITY):
        self.network = build_network()
        self.session = session
        self.batch_size = batch_size
        self.keep_probability = keep_probability

        # Load the validation segment
        self.validation_segment = pickle.load(
            open("pickle-files/preprocessed_validation_segment.p",
                 mode="rb")
        )

    def train_batch(self, batch_x, batch_y):
        session.run(
            self.network.optimizer,
            feed_dict = {
                self.network.x: batch_x,
                self.network.y: batch_y,
                self.network.keep_prob: self.keep_probability
            }
        )

    def evaluate(self, x, y):
        cost = session.run(
            self.network.cost,
            feed_dict = {
                self.network.x: batch_x,
                self.network.y: batch_y,
                self.network.keep_prob: 1.0
            }
        )

        accuracy = session.run(
            self.network.accuracy,
            feed_dict = {
                self.network.x: batch_x,
                self.network.y: batch_y,
                self.network.keep_prob: 1.0
            }
        )

        return (cost, accuracy)

    def log_training_evaluation(self, batch_x, batch_y):
        train_c, train_a = self.evaluate(batch_x, batch_y)
        print(f"Train Cost: {train_c:.3f}"
              f"\tTrain Accu: {train_a:.3f}",
              end="")

    def log_validation_evaluation(self):
        valid_x, valid_y = self.validation_set
        valid_c, valid_a = self.evaluate(valid_x, valid_y)
        print(f"Valid Cost: {valid_c:.3f}"
              f"\tValid Accu: {valid_a:.3f}",
              end="")

    def train_epoch_segment(self, epoch_idx, segment_idx, batches):
        num_batches = len(num_batches)
        batch_logging_mod = int(0.1 * num_batches)
        for batch_idx, batch_x, batch_y in enumerate(batches):
            self.train_batch(batch_x, batch_y)
            if (batch_idx + 1) % batch_logging_mod == 0:
                percent_complete = int(100 * (batch_idx+1) / num_batches)
                print(
                    f"Epoch {epoch_idx:>2} "
                    f"CIFAR-10 Segment {segment_idx:>2} "
                    f"Batch Idx: {batch_idx:>2} "
                    f"%age Complete {percent_complete:>2}: ",
                    end=""
                )
                self.log_training_evaluation(batch_x, batch_y)
                print()

    def train_epoch(self, epoch_idx):
        epoch_start_time = time.time()
        for segment_idx in range(NUM_SEGMENTS):
            batches = load_segment_in_batches(
                segment_idx, self.batch_size
            )
            self.train_epoch_segment(epoch_idx, segment_idx, batches)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print(
            f"Epoch {epoch_idx:>2}: ",
            end=""
        )
        self.log_validation_evaluation(*self.validation_segment)
        print(f" Elapsed Epoch Time: {epoch_time:.1f}sec")
        print()

    def train_epochs(self, num_epochs):
        for epoch_idx in range(num_epochs):
            self.train_epoch(epoch_idx)

def run(session):
    # Initializing the variables
    session.run(tf.global_variables_initializer())

    trainer = Trainer(session)
    trainer.run_epochs(1)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(session, save_model_path)

if __name__ == "__main__":
    with tf.Session() as session:
        run(session)
