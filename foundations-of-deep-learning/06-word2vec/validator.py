import config
import random
import tensorflow as tf

def sample_int_words(vocab_size):
    min_int_word = int(
        config.VALIDATION_WORD_RANGE[0] * vocab_size
    )
    max_int_word = int(
        config.VALIDATION_WORD_RANGE[1] * vocab_size
    )

    sampled_words = []
    for _ in range(config.NUM_VALIDATION_WORDS):
        int_word = random.randrange(min_int_word, max_int_word)
        sampled_words.append(int_word)

    return sampled_words

class Validator:
    def __init__(self, vocab_size, embedding_matrix):
        self.embedding_matrix_ = embedding_matrix
        self.validation_words_ = sample_int_words(vocab_size)
        self.similarity_scores_ = None

    def embedding_matrix(self):
        return self.embedding_matrix_

    def validation_words(self):
        return validation_words_

    def similarity_scores(self):
        if self.similarity_scores_:
            return self.similarity_scores_

        validation_words = tf.constant(
            self.validation_words(),
            dtype=tf.int32
        )
        embedding_representation_norms = (
            tf.sqrt(tf.reduce_sum(tf.square(self.embedding_matrix()), 1))
        )
        normalized_representations = (
            self.embedding_matrix() / embedding_representation_norms
        )

        normalized_validation_representations = tf.nn.embedding_lookup(
            normalized_embedding_matrix, validation_words
        )

        self.similarity_scores_ = tf.matmul(
            normalized_validation_representations,
            tf.transpose(normalized_embedding_matrix)
        )

        return self.similarity_scores_

    def run(self, session, batcher):
        results = {}

        similarity_scores = session.run(self.similarity_scores())
        for idx, int_word in enumerate(self.validation_words()):
            word = batcher.int_to_word(int_word)

            nearest_int_words = (-similarity_scores[idx, :]).argsort()
            # Note that we need to offset by one because the most
            # similar word is the word itself!
            nearest_int_words = (
                nearest_int_words[1:(1 + config.NUM_VALIDATION_RESULTS)]
            )
            nearest_words = map(
                lambda int_word: batcher.int_to_word(int_word),
                nearest_int_words
            )

            results[word] = nearest_words

        return results

    def run_and_log(self, run_info, batch_info):
        ri, bi = run_info, batch_info

        results = self.run(ri.session, ri.batcher)

        print(f"Epoch: {bi.epoch_idx:03d} | "
              f"Batch: {bi.batch_idx:04d} / {ri.batches_per_epoch:04d} | "
              f"Validation!")

        for (word, similar_words) in results:
            similar_words_str = ", ".join(similar_words)
            print(f">> {word}: {similar_words_str}")
