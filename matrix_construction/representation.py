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
    def __init__(self, model: CNN_2D, verbose=False, sparse=False):
        super().__init__()
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.channels = model.channels
        self.act_fn = torch.nn.ReLU()
        self.model = model
        self.verbose = verbose
        self.sparse = sparse

        self.current_output = None #Saves the output of the neural network on the current sample in the forward method

        self.conv_layers = []
        self.fc_layers = []
        self.conv_biases = []
        self.mlp_biases = []

        c, w, w = model.input_shape
        input_shape = (c, w+2, w+2)

        #print(model)

        total = 0

        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):

                matrix = conv2circ(layer,
                                   input_shape,
                                   self.device,
                                   verbose=verbose)[:, :, w+3:-w-3]
                idx = get_indices(matrix.shape[2], w)

                if self.sparse:
                    matrix = matrix[:, :, idx].to_sparse()
                else:
                    matrix = matrix[:, :, idx]

                if self.verbose:
                    memory_bytes = matrix.element_size() * matrix.nelement()
                    memory_gb = memory_bytes / (1024 * 1024 * 1024)
                    total += memory_gb
                    print("Memory usage ConvLayer: {:.5f} GB".format(memory_gb))
                    print("Total: ", total)

                    nonzero = torch.count_nonzero(matrix)
                    a1, a2, a3 = matrix.shape
                    print("Total number of elements: ", a1*a2*a3)
                    print("Percentage of non-zero elements : ", nonzero/(a1*a2*a3))

                self.conv_layers.append(matrix)

                if self.sparse:
                    y = torch.zeros(layer.out_channels*w*w).to(self.device).to_sparse()
                else:
                    y = torch.zeros(layer.out_channels * w * w).to(self.device)

                self.conv_biases.append(y)

            elif isinstance(layer, nn.BatchNorm2d):
                gamma = layer.weight.data.to(self.device)
                beta = layer.bias.data.to(self.device)
                mu = layer.running_mean.to(self.device)
                sigma = layer.running_var.to(self.device)
                epsilon = layer.eps
                channels = gamma.shape[0]
                factor = torch.sqrt(sigma + epsilon).to(self.device)
                weight = (gamma/factor).to(self.device)
                bias = beta - mu * gamma / factor
                bias = bias.repeat(1,w*w).to(self.device)
                matrix = torch.zeros(channels, w * w, channels * w * w).to(self.device)
                idx = (torch.arange(channels).unsqueeze(-1).unsqueeze(-1) * w * w).to(self.device)

                if self.sparse:
                    matrix[:, range(w * w), idx + torch.arange(w * w).to(self.device)] = weight.unsqueeze(-1).unsqueeze(
                        -1).to_sparse()
                else:
                    matrix[:, range(w * w), idx + torch.arange(w * w).to(self.device)] = weight.unsqueeze(-1).unsqueeze(
                        -1)

                self.conv_layers.append(matrix)

                self.conv_biases.append(bias)

                if self.verbose:
                    memory_bytes = matrix.element_size() * matrix.nelement()
                    memory_gb = memory_bytes / (1024 * 1024 * 1024)
                    total += memory_gb
                    print("Memory usage BN2D: {:.5f} GB".format(memory_gb))
                    print("Total: ", total)

                    nonzero = torch.count_nonzero(matrix)
                    a1, a2, a3 = matrix.shape
                    print("Total number of elements: ", a1 * a2 * a3)
                    print("Non-zero elements : ", nonzero)

            elif isinstance(layer, torch.nn.AvgPool2d):
                matrix, dim = avg_pool2d_to_matrix(4,
                                                   (self.channels[-1],
                                                    model.input_shape[1],
                                                    model.input_shape[2]),
                                                    self.device)

                if self.sparse:
                    matrix = matrix.to_sparse()

                self.conv_layers.append(matrix)
                self.conv_biases.append(torch.zeros(dim).to(self.device))

                if self.verbose:
                    memory_bytes = matrix.element_size() * matrix.nelement()
                    memory_gb = memory_bytes / (1024 * 1024 * 1024)
                    total += memory_gb
                    print("Memory usage AvgPooling: {:.5f} GB".format(memory_gb))
                    print("Total: ", total)

            elif isinstance(layer, torch.nn.Linear):
                matrix = layer.weight.data.to(self.device)
                self.fc_layers.append(matrix)

                if model.bias:
                    self.mlp_biases.append(layer.bias.data.to(self.device))
                else:
                    self.mlp_biases.append(torch.zeros(layer.out_features).to(self.device))

        if verbose:
            print("TOTAL MEMORY for representation in GB: ", total)
        self.model_gigas = total

    def forward(self, x):
        self.model.save = True
        self.current_output = self.model(x, rep=True)  # saves activations and preactivations

        flat_x = torch.flatten(x).to(device=self.device)

        if self.sparse:
            a = sparse_multiply_dense(self.conv_layers[0], flat_x[None, None, :])
        else:
            a = self.conv_layers[0] * flat_x[None, None, :]

        A = a.reshape(-1, a.size(-1))
        a = self.conv_biases[0]

        if self.sparse:
            A = sparse_reshape(a, (-1, a.size(-1)))

        if self.verbose:
            memory_bytes = A.element_size() * A.nelement()
            memory_gb = memory_bytes / (1024 * 1024 * 1024)
            print("Memory usage A: {:.5f} GB".format(memory_gb))
            #nonzero = torch.count_nonzero(A)
            #a1, a2, a3 = A.shape
            #print("Total number of elements: ", a1 * a2 * a3)
            #print("Percentage of Non-zero elements : ", nonzero/(a1 * a2 * a3))

        # Conv layers
        for i in range(1, len(self.conv_layers)-1):
            layeri = self.conv_layers[i]

            pre_act = self.model.pre_acts[i-1]
            post_act = self.model.acts[i-1]

            vertices = post_act / pre_act
            vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0
            vertices = torch.flatten(vertices)

            if self.sparse:
                b = sparse_multiply_dense(layeri, vertices[None, None, :])
                B = sparse_reshape(b, (-1, b.size(-1)))
            else:
                b = layeri * vertices[None, None, :]
                B = b.reshape(-1, b.size(-1))

            A = torch.matmul(B, A)

            if self.model.bias or self.model.batch_norm:
                b = torch.Tensor(self.conv_biases[i])

                a = torch.matmul(B, a) + b

            if self.verbose:
                memory_bytes = B.element_size() * B.nelement()
                memory_gb = memory_bytes / (1024 * 1024 * 1024)
                print("Memory usage B: {:.5f} GB".format(memory_gb))
                #nonzero = torch.count_nonzero(B)
                #a1, a2, a3 = B.shape
                ##print("Total number of elements: ", a1 * a2 * a3)
                #print("Non-zero elements : ", nonzero)

                memory_bytes = A.element_size() * A.nelement()
                memory_gb = memory_bytes / (1024 * 1024 * 1024)
                print("Memory usage A: {:.5f} GB".format(memory_gb))
                #nonzero = torch.count_nonzero(A)
                #a1, a2, a3 = A.shape
                #print("Total number of elements: ", a1 * a2 * a3)
                #print("Non-zero elements : ", nonzero)

        # Average pooling layer
        m = len(self.channels)*2 if self.model.batch_norm else len(self.channels)

        if self.sparse:
            layeri = self.conv_layers[m].to_dense()
        else:
            layeri = self.conv_layers[m]

        pre_act = self.model.pre_acts[m - 1]
        post_act = self.model.acts[m - 1]
        vertices = post_act / pre_act
        vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0
        vertices = torch.flatten(vertices)

        b = layeri * vertices[None, None, :]
        B = b.reshape(-1, b.size(-1))

        if self.sparse:
            A = A.to_dense()

        A = torch.matmul(B, A)

        # First FC layer
        # remove the +1 if there is no average pooling
        m = len(self.channels)*2+1 if self.model.batch_norm else len(self.channels)+1

        if self.model.bias or self.model.batch_norm:
            b = self.conv_biases[-1]
            a = torch.matmul(B, a) + b

        vertices = self.model.acts[m-1]/self.model.pre_acts[m-1]
        vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0
        vertices = torch.flatten(vertices)
        B = self.fc_layers[0] * vertices
        A = torch.matmul(B, A)

        if self.model.bias or self.model.batch_norm:
            b = self.mlp_biases[0]
            a = torch.matmul(B, a) + b

        # FC layers
        for i in range(1, len(self.fc_layers)):
            pre_act = self.model.pre_acts[m+i-1]
            post_act = self.model.acts[m+i-1]

            vertices = post_act / pre_act
            vertices[torch.isnan(vertices) | torch.isinf(vertices)] = 0.0

            B = self.fc_layers[i] * vertices
            A = torch.matmul(B, A)

            if self.model.bias or self.model.batch_norm:
                b = self.mlp_biases[i]
                a = torch.matmul(B, a) + b

        if self.model.bias or self.model.batch_norm:
            return torch.cat([A, a.unsqueeze(1)], dim=1)

        else:
            return A


def delete_columns(tensor, k):
    mask = []
    count = 0

    for i in range(tensor.shape[1]):
        count += 1

        if count == k + 1:
            continue
        if count == k + 2:
            count = 0
            continue

        mask.append(i)

    return tensor[:, mask]

def circular_shift(input_tensor: torch.Tensor, num_shifts: int) -> torch.Tensor:
    if num_shifts == 0:
        return input_tensor
    input_size = input_tensor.size(-1)
    num_shifts = num_shifts % input_size
    return torch.cat([input_tensor[..., -num_shifts:], input_tensor[..., :-num_shifts]], dim=-1)

def circulant_matrix(v: torch.Tensor, input_shape: tuple[int,int,int], kernel_size: int, device: str) -> torch.Tensor:
    m = input_shape[1] - kernel_size + 1
    d = [circular_shift(v, j + i * input_shape[1]) for i in range(m) for j in range(m)]
    return torch.stack(d).to(device)

def conv2circ(conv_layer: torch.nn.Module, input_shape: tuple[int,int,int], device: str, verbose=False) -> torch.Tensor:
    """Returns a tensor of shape [out_channels, out_size, input_size],
    each matrix [i,:,:] produces a feature map when applied to the input"""
    weight = conv_layer.weight.data.to(device)
    out_C, in_C, K, _ = weight.shape
    s_range = torch.arange(K, device=device)
    vec_indices = s_range[:, None] * input_shape[1] + torch.arange(K, device=device)
    vec_indices = vec_indices.flatten()
    vec = torch.zeros(out_C, in_C, input_shape[1]*input_shape[2], device=device)
    vec[:, :, vec_indices] = weight.view(out_C, in_C, -1)
    in_circ = [[circulant_matrix(vec[c_out, c_in], input_shape, K, device) for c_in in range(in_C)] for c_out in range(out_C)]
    in_circ = torch.stack([torch.cat(channel_circs, dim=1) for channel_circs in in_circ]).to(device)

    if verbose:
        memory_bytes = in_circ.element_size() * in_circ.nelement()
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        print(f"Memory usage: {memory_gb:.5f} GB")

    return in_circ

def get_indices(s: int, w: int) -> torch.Tensor:
    # Create a single pattern
    pattern = torch.cat([
        torch.ones(w, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
    ]).repeat(w - 1)
    pattern = torch.cat([
        pattern,
        torch.ones(w, dtype=torch.bool),
        torch.zeros(2 * w + 6, dtype=torch.bool),
    ])

    # Calculate the number of patterns that fit in the list
    pattern_len = pattern.size(0)
    num_patterns = (s + pattern_len - 1) // pattern_len

    # Repeat the pattern and trim to the desired length
    bool_list = pattern.repeat(num_patterns)[:s]

    return bool_list

def avg_pool2d_to_matrix(kernel_size: int, input_shape: int, device: str) -> tuple[torch.Tensor, int]:
    # Build matrix representation of average pooling
    # TODO: Should work for any kernel size
    assert kernel_size == 4, f"Hardcoded for kernel size of 4, not {kernel_size}"
    input_channels, input_height, input_width = input_shape
    output_height = input_height // kernel_size
    output_width = input_width // kernel_size

    pooling_matrix = torch.zeros(input_channels * output_height * output_width, input_channels * input_height * input_width, device=device)

    for c in range(input_channels):
        for i in range(output_height):
            for j in range(output_width):
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        row = c * output_height * output_width + i * output_width + j
                        col = c * input_height * input_width + (i * kernel_size + m) * input_width + (j * kernel_size + n)
                        pooling_matrix[row, col] = 1 / (kernel_size * kernel_size)

    return pooling_matrix, input_channels * output_height * output_width

def sparse_reshape(sparse_tensor: torch.Tensor, new_shape: tuple[int,int]) -> torch.Tensor:
    if len(new_shape) != 2:
        raise ValueError("Only 2D tensors are supported for sparse reshaping.")

    old_shape = sparse_tensor.shape
    if -1 in new_shape:
        new_shape = list(new_shape)
        new_shape[new_shape.index(-1)] = old_shape[0] * old_shape[1] // abs(new_shape[new_shape.index(-1)])
        new_shape = tuple(new_shape)

    #if old_shape[0] * old_shape[1] != new_shape[0] * new_shape[1]:
    #    raise ValueError("The new shape must have the same number of elements as the original shape.")

    coo_data = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()

    # Compute new coordinates
    old_row_indices = coo_data[0, :]
    old_col_indices = coo_data[1, :]

    linear_indices = old_row_indices * old_shape[1] + old_col_indices
    new_row_indices = linear_indices // new_shape[1]
    new_col_indices = linear_indices % new_shape[1]

    new_indices = torch.stack([new_row_indices, new_col_indices])

    # Create the new sparse tensor
    reshaped_sparse_tensor = torch.sparse_coo_tensor(new_indices, values, new_shape, dtype=sparse_tensor.dtype,
                                                     device=sparse_tensor.device)

    return reshaped_sparse_tensor

def sparse_multiply_dense(sparse_tensor: torch.Tensor, dense_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure that dense_tensor has 3 dimensions
    if dense_tensor.dim() != 3:
        raise ValueError("dense_tensor must have 3 dimensions")

    # Calculate the broadcasted shape
    broadcast_shape = torch.Size([
        max(sparse_tensor.shape[0], dense_tensor.shape[0]),
        max(sparse_tensor.shape[1], dense_tensor.shape[1]),
        max(sparse_tensor.shape[2], dense_tensor.shape[2])
    ])

    # Calculate the new indices and values of the resulting sparse tensor after the element-wise multiplication
    coo_data = sparse_tensor.coalesce().indices()
    values = sparse_tensor.coalesce().values()
    new_indices = []
    new_values = []

    for idx, value in zip(coo_data.t(), values):
        dense_value = dense_tensor[0, 0, idx[2]]
        new_value = value * dense_value

        if not torch.all(new_value == 0):
            new_indices.append(idx.unsqueeze(0))
            new_values.append(new_value)

    new_indices = torch.cat(new_indices, dim=0).t()
    new_values = torch.stack(new_values)

    # Create a new sparse tensor with the calculated indices, values, and the broadcasted shape
    return torch.sparse_coo_tensor(new_indices, new_values, broadcast_shape, dtype=sparse_tensor.dtype,
                                                   device=sparse_tensor.device)
