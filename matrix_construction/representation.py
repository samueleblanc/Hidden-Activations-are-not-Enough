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
            vertices = torch.where(torch.isnan(vertices) | torch.isinf(vertices), torch.tensor(0.0).to(self.device), vertices)

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
    def __init__(self, model: CNN_2D, batchsize=32, device=None) -> None:
        super().__init__()
        self.device = 'cpu' if device is None else device
        self.model = model
        self.batchsize = batchsize
        self.channels = model.channels
        self.act_fn = model.get_activation_fn()
        # Collect layers in processing order
        self.layers = []
        self.layers.extend(model.conv_layers)
        self.layers.extend(model.fc_layers)
        self.in_c, self.in_h, self.in_w = model.input_shape
        self.input_size: int = self.in_c * self.in_h * self.in_w
        self.current_output: CNN_2D | None = None

        # Find first FC layer index
        self.first_fc_layer = None
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                self.first_fc_layer = i
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.model.save = True
            print(x.shape)
            print(x.unsqueeze(0).shape)
            self.model.train(False)
            print(self.model.training)
            self.current_output = self.model(x, rep=True)

            # Total number of positions and batches needed
            C, H, W = self.in_c, self.in_h, self.in_w
            total_positions = C * H * W
            num_batches = (total_positions + self.batchsize - 1) // self.batchsize

            A = torch.Tensor().to(self.device)

            for batch_idx in range(num_batches):
                # Calculate batch indices
                start = batch_idx * self.batchsize
                end = min((batch_idx + 1) * self.batchsize, total_positions)
                current_batch_size = end - start

                # Create indices for this batch
                indices = torch.arange(start, end, device=self.device)
                c = indices // (H * W)
                remaining = indices % (H * W)
                h = remaining // W
                w = remaining % W

                # Create batched input for this chunk
                batched_input = torch.zeros((current_batch_size, C, H, W), device=self.device)
                batched_input[torch.arange(current_batch_size), c, h, w] = x.flatten()[start:end]

                B = batched_input
                layer_idx = 0

                for i, layer in enumerate(self.layers):
                    if not isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.Linear, nn.BatchNorm2d)):
                        continue

                    if layer_idx >= len(self.model.pre_acts):
                        break

                    # Get activation ratios
                    pre_act = self.model.pre_acts[layer_idx]
                    post_act = self.model.acts[layer_idx]
                    vertices = post_act / pre_act
                    vertices = torch.where(
                        torch.isnan(vertices) | torch.isinf(vertices),
                        torch.tensor(0.0, device=self.device),
                        vertices
                    ).squeeze(0)  # Remove original batch dim

                    if isinstance(layer, nn.Conv2d):
                        B = layer(B)
                        # Expand vertices to match current batch size
                        B = B * vertices.repeat(current_batch_size, 1, 1, 1)
                        layer_idx += 1
                    elif isinstance(layer, nn.AvgPool2d):
                        B = layer(B)
                        layer_idx += 1
                    elif isinstance(layer, nn.BatchNorm2d):
                        B = layer(B)
                        scale = (layer.weight.data / torch.sqrt(layer.running_var + layer.eps))
                        B = B * scale.view(1, -1, 1, 1)
                        layer_idx += 1
                    elif isinstance(layer, nn.Linear):
                        if i == self.first_fc_layer:
                            B = B.view(B.size(0), -1)
                            B = torch.matmul(layer.weight.data, B.T).T
                            B = B * vertices.view(1, -1).repeat(current_batch_size, 1)
                        else:
                            # SPECIAL HANDLING FOR LAST LAYER, which is not working yet...
                            if layer == self.layers[-1]:  # Final classification layer
                                B = torch.matmul(layer.weight.data, B.T).T
                                if self.model.bias:
                                    B += layer.bias.data
                            else:
                                B = torch.matmul(layer.weight.data, B.T).T
                                B = B * vertices.view(1, -1).repeat(current_batch_size, 1)
                        layer_idx += 1

                # Accumulate results
                A = torch.cat((A, B.T), dim=1) if A.numel() else B.T

            # Compute bias term
            if self.model.bias or self.model.batch_norm:
                a = torch.zeros_like(x).unsqueeze(0).to(self.device)
                layer_idx = 0
                for i, layer in enumerate(self.layers):
                    if not isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.Linear, nn.BatchNorm2d)):
                        continue

                    if layer_idx >= len(self.model.pre_acts):
                        break

                    pre_act = self.model.pre_acts[layer_idx]
                    post_act = self.model.acts[layer_idx]
                    vertices = post_act / pre_act
                    vertices = torch.where(
                        torch.isnan(vertices) | torch.isinf(vertices),
                        torch.tensor(0.0, device=self.device),
                    ).squeeze(0)

                    if isinstance(layer, nn.Conv2d):
                        a = layer(a)
                        a = a * vertices
                        layer_idx += 1
                    elif isinstance(layer, nn.BatchNorm2d):
                        a = layer(a)
                        scale = (layer.weight.data / torch.sqrt(layer.running_var + layer.eps))
                        a = a * scale.view(1, -1, 1, 1)
                        layer_idx += 1
                    elif isinstance(layer, nn.Linear):
                        a = a.view(1, -1)
                        a = a * vertices.view(1, -1)
                        a = torch.matmul(layer.weight.data, a.T).T
                        if self.model.bias:
                            a += layer.bias.data
                        layer_idx += 1

                return torch.cat((A, a.squeeze(0).unsqueeze(-1)), dim=1)
            else:
                return A


if __name__ == "__main__":
    torch.manual_seed(41)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    #device = 'cpu'
    print(f"Using device: {device}")

    x = torch.rand((3,12,12)).to(device)
    cnn = CNN_2D(input_shape=(3,12,12), num_classes=10, channels=(8, 16), fc=(100), kernel_size=(3, 3),
        bias=False, batch_norm=False, dropout=False, activation="relu").to(device)
    forward_pass = cnn(x)
    rep = ConvRepresentation_2D(cnn, device=device)
    rep = rep.forward(x)
    one = torch.flatten(torch.ones(cnn.matrix_input_dim)).to(device)
    rep_forward = torch.matmul(rep, one)
    print(forward_pass)
    print(rep_forward)
    diff = torch.norm(rep_forward - forward_pass).item()
    # 0.00749 on cpu float precision
    print(diff)


