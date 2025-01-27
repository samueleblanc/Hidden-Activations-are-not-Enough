"""
    Given an MLP or a CNN it constructs a quiver representation with the bias and weights, and computes with the forward method
    a matrix obtained by multiplying all matrices in the induced quiver representation \phi(W,f)(x)
"""

import torch
import torch.nn as nn

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D


class MlpRepresentation:
    def __init__(self, model: MLP, device="cpu") -> None:
        # TODO: Optimize this algorithm like for the one with convolutions
        self.device = device
        self.act_fn = model.get_activation_fn()()
        self.mlp_weights = []
        self.mlp_biases = []
        self.input_size = model.input_size
        self.model = model

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                self.mlp_weights.append(layer.weight.data)

                if model.bias:
                    self.mlp_biases.append(layer.bias.data)
                else:
                    self.mlp_biases.append(torch.zeros(layer.out_features))

            elif isinstance(layer, nn.BatchNorm1d):
                gamma = layer.weight.data
                beta = layer.bias.data
                mu = layer.running_mean
                sigma = layer.running_var
                epsilon = layer.eps

                factor = torch.sqrt(sigma + epsilon)

                self.mlp_weights.append(torch.diag(gamma/factor))
                self.mlp_biases.append(beta - mu*gamma/factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_x = torch.flatten(x).to(device=self.device)
        self.model.save = True
        _ = self.model(flat_x)

        A = self.mlp_weights[0].to(self.device) * flat_x

        a = self.mlp_biases[0]

        for i in range(1, len(self.mlp_weights)):
            layeri = self.mlp_weights[i].to(self.device)

            pre_act = self.model.pre_acts[i-1]
            post_act = self.model.acts[i-1]

            vertices = post_act / pre_act
            vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0

            B = layeri * vertices
            A = torch.matmul(B, A)

            if self.model.bias or self.model.batch_norm:
                b = self.mlp_biases[i]
                a = torch.matmul(B, a) + b

        if self.model.bias or self.model.batch_norm:
            return torch.cat([A, a.unsqueeze(1)], dim=1)

        else:
            return A


class ConvRepresentation_2D:
    def __init__(self, model: CNN_2D) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.act_fn = model.get_activation_fn()
        self.layers = list(model.modules())
        self.in_c, self.in_h, self.in_w = model.input_shape
        self.input_size: int = self.in_c*self.in_h*self.in_w

        self.current_output: CNN_2D|None = None  #Saves the output of the neural network on the current sample in the forward method

        for i,layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                self.first_fc_layer = i
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Parallelize the forward passes
        with torch.no_grad():
            self.model.save = True
            self.current_output = self.model(x, rep=True)  # Saves activations and preactivations
            # TODO: We should be able to save activations and preacts for a NN that we get online

            A = torch.Tensor()  # Will become M(W,f)(x)
            zeros = torch.zeros((1,self.in_c,self.in_h,self.in_w)).to(self.device)  # Input used to compute the columns of M(W,f)(x)
            for c in range(self.in_c):
                for h in range(self.in_h):
                    for w in range(self.in_w):
                        zeros[0][c][h][w] = x[c][h][w].item()
                        i = 0
                        max_pool = -9
                        for j,layer in enumerate(self.layers):
                            pre_act = self.model.pre_acts[i-1]
                            post_act = self.model.acts[i-1]
                            vertices = post_act / pre_act
                            vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0
                            if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
                                if i == 0:
                                    B = layer(zeros).to(self.device)
                                else:
                                    B = B * vertices
                                    B = layer(B)
                                i += 1
                            elif isinstance(layer, nn.Linear):
                                if j == self.first_fc_layer:
                                    if max_pool != i-1: B = B * vertices
                                    B = torch.matmul(layer.weight.data, B.view(-1).unsqueeze(-1))
                                else:
                                    B = B * vertices.unsqueeze(-1)
                                    B = torch.matmul(layer.weight.data, B)
                                i += 1
                            elif isinstance(layer, nn.BatchNorm2d):
                                B = B * vertices
                                B = B * (layer.weight.data/torch.sqrt(layer.running_var+layer.eps)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                                i += 1
                            elif isinstance(layer, nn.MaxPool2d):
                                max_pool = i
                                B = B * vertices
                                pool = self.model.acts[i]
                                batch_indices = torch.arange(pool.shape[0]).view(-1,1,1,1)
                                channel_indices = torch.arange(pool.shape[1]).view(1,-1,1,1)
                                row_indices = pool // B.shape[2]
                                col_indices = pool % B.shape[3]
                                B = B[batch_indices, channel_indices, row_indices, col_indices]
                                i += 1

                        A = torch.cat((A,B),dim=-1)  # Cat the vector produced to the matrix M(W,f)(x)
                        zeros[0][c][h][w] = 0.0

            if self.model.bias or self.model.batch_norm:
                i = 0
                max_pool = -9
                for j,layer in enumerate(self.layers):
                    pre_act = self.model.pre_acts[i-1]
                    post_act = self.model.acts[i-1]
                    vertices = post_act / pre_act
                    vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0
                    if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
                        if i == 0:
                            a = torch.zeros(x.shape).to(self.device)
                        else:
                            a = a * vertices
                        a = layer(a)
                        i += 1
                    elif isinstance(layer, nn.Linear):
                        if j == self.first_fc_layer:
                            if max_pool != i-1: a = a * vertices
                            a = torch.matmul(layer.weight.data, a.view(-1).unsqueeze(-1))
                        else:
                            a = a * vertices.unsqueeze(-1)
                            a = torch.matmul(layer.weight.data, a)
                        if self.model.bias: a = a + layer.bias.data.unsqueeze(-1)
                        i += 1
                    elif isinstance(layer, nn.BatchNorm2d):
                        a = a * vertices
                        a = layer(a)
                        i += 1
                    elif isinstance(layer, nn.MaxPool2d):
                        max_pool = i
                        a = a * vertices
                        pool = self.model.acts[i]
                        batch_indices = torch.arange(pool.shape[0]).view(-1,1,1,1)
                        channel_indices = torch.arange(pool.shape[1]).view(1,-1,1,1)
                        row_indices = pool // a.shape[2]
                        col_indices = pool % a.shape[3]
                        a = a[batch_indices, channel_indices, row_indices, col_indices]
                        i += 1

                return torch.cat((A, a), dim=1)

            else:
                return A
