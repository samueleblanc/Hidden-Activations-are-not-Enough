"""
    Given an MLP or a CNN it constructs a quiver representation with the bias and weights, and computes with the forward method
    a matrix obtained by multiplying all matrices in the induced quiver representation \phi(W,f)(x)
"""

import torch
import torch.nn as nn

from model_zoo.mlp import MLP
from model_zoo.cnn import CNN_2D
from model_zoo.res_net import ResNet


class MlpRepresentation:
    def __init__(self, model: MLP, build_rep:bool=False, device:bool="cpu") -> None:
        # TODO: Optimize this algorithm like for the one with convolutions
        self.device = device
        self.act_fn = model.get_activation_fn()()
        self.mlp_weights = []
        self.mlp_biases = []
        self.input_size = model.input_size
        self.model = model
        self.build_rep = build_rep  # TODO: Currently doesn't work if the model has biases

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
        if self.build_rep: self.rep = [A + a.unsqueeze(-1)]

        for i in range(1, len(self.mlp_weights)):
            layeri = self.mlp_weights[i].to(self.device)

            pre_act = self.model.pre_acts[i-1]
            post_act = self.model.acts[i-1]

            vertices = post_act / pre_act
            vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0

            B = layeri * vertices
            if self.build_rep: self.rep.append(B)
            A = torch.matmul(B, A)

            if self.model.bias or self.model.batch_norm:
                b = self.mlp_biases[i]
                a = torch.matmul(B, a) + b
                if self.build_rep: self.rep[-1] = torch.cat((self.rep[-1], b.unsqueeze(-1)), dim=1)

        if self.model.bias or self.model.batch_norm:
            return torch.cat([A, a.unsqueeze(1)], dim=1)
        else:
            return A


class ConvRepresentation_2D:
    def __init__(self, model: CNN_2D|ResNet, batch_size:int=1) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.batch_size = batch_size
        self.act_fn = model.get_activation_fn()
        self.layers = list(model.conv_layers + model.fc_layers)
        self.in_c, self.in_h, self.in_w = model.input_shape
        self.input_size: int = self.in_c*self.in_h*self.in_w

        self.current_output: CNN_2D|ResNet|None = None  # Saves the output of the NN on the current sample in the forward method

        # Find index of first FC layer (used for flatten layer)
        for i,layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                self.first_fc_layer = i
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model.save = True
            self.current_output = self.model(x, rep=True)  # Saves activations and preactivations

            # Total number of positions and batches needed
            C, H, W = self.in_c, self.in_h, self.in_w
            total_positions = C*H*W
            num_batches = (total_positions + self.batch_size - 1)//self.batch_size

            A = torch.Tensor().to(self.device)  # Will become the matrix M(W,f)(x)

            for batch in range(num_batches):
                # Compute batch indices
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, total_positions)
                current_batch_size = end - start

                # Create indices for this batch
                indices = torch.arange(start, end, device=self.device)
                c = indices // (H*W)
                remaining = indices % (H*W)
                h = remaining // W
                w = remaining % W

                # Create batched input for this chunk
                batched_input = torch.zeros((current_batch_size,C,H,W), device=self.device)
                batched_input[torch.arange(current_batch_size),c,h,w] = x.flatten()[start:end]
                
                B = batched_input
                i = 0
                max_pool = -9
                for j,layer in enumerate(self.layers):
                    # Get activation ratios
                    pre_act = self.model.pre_acts[i-1]
                    post_act = self.model.acts[i-1]
                    vertices = post_act / pre_act
                    vertices = torch.where(
                        torch.isnan(vertices) | torch.isinf(vertices),
                        torch.tensor(0.0, device=self.device),
                        vertices
                    ).squeeze(0)  # Remove original batch dim

                    if isinstance(layer, nn.Conv2d):
                        if i != 0 and max_pool != i-1: B = B * vertices.repeat(current_batch_size,1,1,1)
                        B = layer(B)
                        i += 1
                    elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                        if max_pool != i-1: B = B * vertices
                        B = layer(B)
                        i += 1
                    elif isinstance(layer, nn.Linear):
                        if j == self.first_fc_layer:
                            B = B.view(B.shape[0], -1)
                        if max_pool != i-1: B = B * vertices.view(1,-1).repeat(current_batch_size,1)
                        B = torch.matmul(layer.weight.data, B.T).T
                        i += 1
                    elif isinstance(layer, nn.BatchNorm2d):
                        if max_pool != i-1: B = B * vertices.repeat(current_batch_size,1,1,1)
                        B = B * (layer.weight.data/torch.sqrt(layer.running_var+layer.eps)).view(1,-1,1,1)
                        i += 1
                    elif isinstance(layer, nn.MaxPool2d):
                        max_pool = i
                        B = B * vertices.repeat(current_batch_size,1,1,1)
                        pool = self.model.acts[i]
                        batch_indices = torch.arange(current_batch_size).view(-1,1,1,1)
                        channel_indices = torch.arange(pool.shape[1]).view(1,-1,1,1)
                        row_indices = pool // B.shape[2]
                        col_indices = pool % B.shape[3]
                        B = B[batch_indices, channel_indices, row_indices, col_indices]
                        i += 1

                # Cat the vector produced to the matrix M(W,f)(x)
                A = torch.cat((A,B.T),dim=-1) if A.numel() else B.T

            if self.model.bias or self.model.batch_norm:
                i = 0
                max_pool = -9
                for j,layer in enumerate(self.layers):
                    pre_act = self.model.pre_acts[i-1]
                    post_act = self.model.acts[i-1]
                    vertices = post_act / pre_act
                    vertices = torch.where(
                        torch.isnan(vertices) | torch.isinf(vertices),
                        torch.tensor(0.0, device=self.device),
                        vertices
                    )
                    if isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
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
