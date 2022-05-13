import os

import numpy as np


def verify_file_exists(path):
    assert os.path.isfile(path+'.npy'), f'File Error: {path}.npz not found!'

class TwoDigitDatasetGenerator:
    """Generate a dataset of two, two digit integers, with additional operator and output variables
    """
    def __init__(self, fp:str='tmp/digit-data/'):
        self.modulo = 100
        self.file_path = fp

    def generate_add_set(self):
        name = 'simple_add'

        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)
        if not os.path.isfile(self.file_path + name):
            print(f'Generating data file {name}')
            arr = []
            for i in range(self.modulo):
                for j in range(self.modulo):
                    if i + j < self.modulo:  # check for two digit output
                        arr.append([i, j, i + j, 0])
            arr_ = np.array(arr)
            np.save(self.file_path+name, arr_)

        verify_file_exists(self.file_path + name)

    def generate_mul_set(self):
        name = 'simple_mul'

        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)
        if not os.path.isfile(self.file_path + name):
            print(f'Generating data file {name}')
            arr = []
            for i in range(self.modulo):
                for j in range(self.modulo):
                    if i * j < self.modulo:  # check for two digit output
                        arr.append([i, j, i * j, 1])
            arr_ = np.array(arr)
            np.save(self.file_path + name, arr_)

        verify_file_exists(self.file_path + name)


c = TwoDigitDatasetGenerator()

c.generate_mul_set()
c.generate_add_set()