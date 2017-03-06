from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DLProgress(tqdm):
    last_block = 0

    def __init__(self):
        super().__init__(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc="CIFAR-10 Dataset"
        )

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

def run():
    if not isfile('cifar-10-python.tar.gz'):
        with DLProgress() as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'cifar-10-python.tar.gz',
                pbar.hook)

    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open('cifar-10-python.tar.gz') as tar:
            tar.extractall()
            tar.close()

if __name__ == "__main__":
    run()
