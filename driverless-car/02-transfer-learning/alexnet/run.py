import bottleneck
import dataset as dataset_mod
import network as network_mod
import pickle
import train as train_mod

def transform():
    dataset = dataset_mod.load_cifar10_dataset()
    network = network_mod.build_alexnet_network(dataset.num_classes)

    bottleneck.transform(dataset, network)

def train():
    with open("../data/bottleneck_alexnet_cifar10.p", "rb") as f:
        dataset = pickle.load(f)
        network = network_mod.build_alexnet_network(dataset.num_classes)
        train_mod.train(dataset, network)

if __name__ == "__main__":
    train()
