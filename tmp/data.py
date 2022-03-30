import os
import tarfile

from typing import Tuple, List
import torch as T


class DataLoader(T.utils.data.Dataset):

    def __init__(self):
        self.cache_dir = []
        self.file_path = 'D:/GitHub/DeepLeaning/mathematics_dataset/' # located in parent directory to repository
        self.file_name = 'mathematics_dataset-v1.0.tar.gz'
        self.cache_dir = 'math_data/'

    def download_datatset(self):
        """Download tar.gz file into local directory, 'tmp/math_data/' (called with `self.cache_dir`)

        :notes  This only needs to be called once if the dataset hasn't already been downloaded
        """
        print('Downloading dataset file into local cache...')
        if not os.path.isdir(self.file_path):
            if not os.path.isfile(os.path.join(self.file_path, self.file_name)):
                assert False, f"Download or move data file into the repository's parent directory {self.file_path}"
            else:
                assert False, f"Can not locate dataset older {self.file_path}. Make sure to configure to personal device."
        else:
            with tarfile.open(os.path.join(self.file_path, self.file_name), "r") as tf:
                tf.extractall(path=self.cache_dir)
        print('Done...')

    def load_file(self, path: str) -> Tuple[List[str], List[str]]:
        print(f"Loading {path}")
        print(os.path.isfile(path))
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines()]

        q = lines[::2]
        a = lines[1::2]
        assert len(q) == len(a)
        return q, a

c = DataLoader()
#c.download_dataset() # download data

q, a = c.load_file(c.cache_dir +  'mathematics_dataset-v1.0/train-easy/arithmetic__mul.txt')

print(q)