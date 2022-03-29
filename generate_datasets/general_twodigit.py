import os


def verify_file_exists(path):
    assert os.path.isfile(path), f'File Error: {path} not found!'


class TwoDigitDatasetGenerator:
    """Generate a dataset of two, two digit integers, with additional operator and output variables
    """
    def __init__(self, operation:int, fp:str='tmp/digit-data/'):
        self.operation = operation # 0 for add, 1 for multiply
        self.modulo = 100

        self.file_path = fp



    def generate_add_set(self):
        name = 'simple_add.txt'

        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)
        if not os.path.isfile(self.file_path + name):
            print(f'Generating data file {name}')
            with open(self.file_path + name, 'w') as f:
                for i in range(self.modulo):
                    for j in range(self.modulo):
                        if i * j < self.modulo:  # check for two digit output
                            f.write(str([[i, j], i + j, 0]))

        verify_file_exists(self.file_path + name)

    def generate_mul_set(self):
        data = []
        name = 'simple_mul.txt'

        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path)
        if not os.path.isfile(self.file_path + name):
            print(f'Generating data file {name}')
            with open(self.file_path + name, 'w') as f:
                for i in range(self.modulo):
                    for j in range(self.modulo):
                        if i * j < self.modulo:  # check for two digit output
                            f.write(str([[i, j], i * j, 1]))

        verify_file_exists(self.file_path + name)