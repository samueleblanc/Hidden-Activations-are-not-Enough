import torch
from model_zoo.alex_net import AlexNet
from model_zoo.res_net import ResNet
from model_zoo.vgg import VGG
from matrix_construction.matrix_computation import ConvRepresentation_2D
import time


if __name__ == '__main__':
    input_shape = (3,224,224)
    print('1')
    num_classes = 10
    model = VGG(input_shape=input_shape,
                num_classes=num_classes,
                pretrained=False).to('mps')
    print('2')
    representation = ConvRepresentation_2D(model=model)
    print('3')
    x = torch.rand(input_shape, device='mps')
    print(x.shape)
    start = time.time()
    know_mat = representation.forward(x)
    print("Time: ", time.time()-start)


