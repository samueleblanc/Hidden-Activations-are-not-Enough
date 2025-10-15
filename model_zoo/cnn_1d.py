from knowledgematrix.neural_net import NN


class CNN1D(NN):

    def __init__(
            self, 
            input_shape: tuple[int],
            num_classes: int,
            save: bool=False, 
            device: str="cpu"
        ) -> None:
        super().__init__(input_shape, save, device)

        self.conv1d(self.input_shape[0], 32, kernel_size=5, stride=1, padding=2)
        self.batchnorm1d(32)
        self.relu()
        self.maxpool1d(2)

        self.conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.batchnorm1d(64)
        self.relu()
        self.maxpool1d(2)

        self.conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm1d(128)
        self.relu()
        self.maxpool1d(2)

        self.adaptiveavgpool1d(3)

        self.flatten()

        self.linear(in_features=128*3, out_features=256)
        self.relu()
        self.dropout(0.5)
        self.linear(in_features=256, out_features=num_classes)