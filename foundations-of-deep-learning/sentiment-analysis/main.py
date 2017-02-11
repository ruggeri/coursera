from input_reader import InputReader
from neural_network.runner import Runner
from test_data_generator import TestDataGenerator
from vocabulary import Vocabulary

def main():
    print("Reading Input")
    (reviews, targets) = InputReader.run()

    print("Building Vocabulary")
    vocabulary = Vocabulary(reviews)
    print(f"Vocab size: {vocabulary.num_words}")
    print("Converting inputs")
    inputs = vocabulary.featurize(reviews)

#    (inputs, targets) = TestDataGenerator(1000).gen_samples(10000)

    print("Beginning Training")
    runner = Runner(inputs, targets)
    for epoch in range(100):
        runner.run_epoch()

if __name__ == "__main__":
    main()
