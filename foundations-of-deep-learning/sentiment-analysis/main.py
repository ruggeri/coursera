from input_reader import InputReader
from neural_network.runner import Runner
from vocabulary import Vocabulary

def main():
    print("Reading Input")
    (reviews, targets) = InputReader.run()

    # Uncomment this for faster testing.
    #(reviews, targets) = (reviews[:100], targets[:100])

    print("Building Vocabulary")
    vocabulary = Vocabulary(reviews, targets)
    print(f"Vocab size: {vocabulary.num_words}")
    print("Converting inputs")
    inputs = vocabulary.featurize(reviews)

    print("Beginning Training")
    runner = Runner(inputs, targets)
    for epoch in range(100):
        runner.run_epoch()

if __name__ == "__main__":
    main()
