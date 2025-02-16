"""
    Implementation for (almost) arbitrary CNN.
    This class has a forward method that can be used for training or to 
    save activations and preactivations to later compute the matrix.
"""

import torch
import torch.nn as nn
import math


class CNN_2D(nn.Module):
    def __init__(
        self, 
        input_shape: tuple[int,int,int], 
        num_classes: int, 
        channels:tuple[int] = (8, 16), 
        padding:tuple[(int,int)] = ((1,1),(1,1)),
        fc:tuple[int] = (500), 
        kernel_size:tuple[(int,int)] = ((3, 3),(3, 3)), 
        bias:bool = False, 
        residual:list[(int,int)] = [],
        batch_norm:bool = False,
        dropout:bool = False,
        activation:str = "relu",
        pooling:str = "avg",
        save:bool=False
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        shape_per_layer = []
        h1 = h
        w1 = w
        for i,ch in enumerate(channels):  # Suppose dilation = stride = 1
            shape_per_layer.append((ch, h1, w1))
            h1 = (h1 + 2*padding[i][0] - 1*(kernel_size[i][0] - 1) - 1)//1 + 1
            w1 = (w1 + 2*padding[i][1] - 1*(kernel_size[i][1] - 1) - 1)//1 + 1

        self.input_shape = input_shape
        self.save = save
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.residual = {a : b for a,b in list(set(residual)) if a < b and shape_per_layer[a] == shape_per_layer[b]}
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout = True
        self.matrix_input_dim = c*w*h + 1 if bias or batch_norm else c*w*h
        self.activation = activation
        self.pooling = pooling

        self.conv_layers.append(nn.Conv2d(
                                    in_channels = c,
                                    out_channels = channels[0] if isinstance(channels, tuple) else channels,
                                    kernel_size = kernel_size[0],
                                    padding = padding[0],
                                    bias = False
                                ))
        
        if batch_norm: self.conv_layers.append(nn.BatchNorm2d(channels[0] if isinstance(channels, tuple) else channels))

        if isinstance(channels, tuple):
            for i in range(1, len(channels)):
                self.conv_layers.append(self.get_activation_fn())
                self.conv_layers.append(nn.Conv2d(
                                            in_channels = channels[i-1],
                                            out_channels = channels[i],
                                            kernel_size = kernel_size[i],
                                            padding = padding[i],
                                            bias = False
                                        ))
                if batch_norm: self.conv_layers.append(nn.BatchNorm2d(channels[i]))

        if self.dropout: self.conv_layers.append(nn.Dropout(0.25))
        self.conv_layers.append(self.get_activation_fn())

        ker_size = 4
        stride = ker_size
        pad = 0
        if pooling == "avg":
            self.conv_layers.append(nn.AvgPool2d(
                                        kernel_size = ker_size, 
                                        padding = pad, 
                                        ceil_mode = False
                                    ))
            self.fc_layers.append(nn.Linear(
                                    in_features = shape_per_layer[-1][0] * self.round_up((shape_per_layer[-1][1]+2*pad-ker_size)/stride + 1) * self.round_up((shape_per_layer[-1][2]+2*pad-ker_size)/stride + 1),
                                    # This is for no average pooling after the last convolution
                                    #in_features=channels[-1] * h * w,
                                    out_features = fc[0] if isinstance(fc, tuple) else fc,
                                    bias = bias
                                 ))
        elif pooling == "max":
            self.conv_layers.append(nn.MaxPool2d(
                                        kernel_size = ker_size, 
                                        padding = pad, 
                                        stride = stride, 
                                        return_indices = True
                                    ))
            self.fc_layers.append(nn.Linear(
                                    in_features = shape_per_layer[-1][0] * self.round_up((shape_per_layer[-1][1]+2*pad-ker_size)/stride + 1) * self.round_up((shape_per_layer[-1][2]+2*pad-ker_size)/stride + 1),
                                    # This is for no average pooling after the last convolution
                                    #in_features=channels[-1] * h * w,
                                    out_features = fc[0] if isinstance(fc, tuple) else fc,
                                    bias = bias
                                  ))

        if isinstance(fc, tuple):
            for i in range(1, len(fc)):
                self.fc_layers.append(self.get_activation_fn())
                if dropout: self.fc_layers.append(nn.Dropout(0.5))
                self.fc_layers.append(nn.Linear(
                                        in_features = fc[i-1],
                                        out_features = fc[i],
                                        bias = bias))
        self.fc_layers.append(self.get_activation_fn())
        # if dropout: self.fc_layers.append(nn.Dropout(0.5))
        self.fc_layers.append(nn.Linear(
                                in_features = fc[-1] if isinstance(fc, tuple) else fc,
                                out_features = num_classes,
                                bias = bias))

    def forward(self, x: torch.Tensor, rep=False) -> torch.Tensor:
        if not rep:
            if self.batch_norm and (len(x.shape) == 3): x = x.unsqueeze(0)
            cnt = 0
            x_res = {}
            if cnt in self.residual: x_res[self.residual[cnt]] = x
            for layer in self.conv_layers:
                if isinstance(layer, nn.Conv2d): 
                    cnt += 1
                elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                    if cnt in x_res:
                        x = x + x_res[cnt]
                        del x_res[cnt]
                    if cnt in self.residual:
                        if self.residual[cnt] not in x_res:
                            x_res[self.residual[cnt]] = x
                        else:
                            x_res[self.residual[cnt]] += x
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                x = torch.flatten(x)
            for layer in self.fc_layers:
                if isinstance(layer, nn.Linear): 
                    cnt += 1
                elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                    if cnt in x_res:
                        x = x + x_res[cnt]
                        del x_res[cnt]
                    if cnt in self.residual:
                        if self.residual[cnt] not in x_res:
                            x_res[self.residual[cnt]] = x
                        else:
                            x_res[self.residual[cnt]] += x
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
                    x = layer(x)
            return x

        self.pre_acts: list[torch.Tensor] = []
        self.acts: list[torch.Tensor] = []
        cnt = 0
        x_res = {}

        x = x.unsqueeze(0)
        if cnt in self.residual: x_res[self.residual[cnt]] = x
        x = self.conv_layers[0](x)  # Conv
        cnt += 1

        if self.save:
            self.pre_acts.append(x.detach().clone())
            if self.batch_norm:
                self.acts.append(x.detach().clone())
        if self.batch_norm:
            x = self.conv_layers[1](x)  # BN
            if self.save:
                self.pre_acts.append(x.detach().clone())
            if cnt in x_res:
                x = x + x_res[cnt]
                del x_res[cnt]
            if cnt in self.residual:
                if self.residual[cnt] not in x_res:
                    x_res[self.residual[cnt]] = x
                else:
                    x_res[self.residual[cnt]] += x
            x = self.conv_layers[2](x)  # Activation
            if self.save:
                self.acts.append(x.detach().clone())
        else:
            if cnt in x_res:
                x = x + x_res[cnt]
                del x_res[cnt]
            if cnt in self.residual:
                if self.residual[cnt] not in x_res:
                    x_res[self.residual[cnt]] = x
                else:
                    x_res[self.residual[cnt]] += x
            x = self.conv_layers[1](x)  # Activation
            if self.save:
                self.acts.append(x.detach().clone())

        for i in range(3 if self.batch_norm else 2, len(self.conv_layers)):
            layer = self.conv_layers[i]
            if isinstance(layer, nn.Conv2d):
                cnt += 1
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())
                    if self.batch_norm:
                        self.acts.append(x.detach().clone())

            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                if cnt in x_res:
                    x = x + x_res[cnt]
                    del x_res[cnt]
                if cnt in self.residual:
                    if self.residual[cnt] not in x_res:
                        x_res[self.residual[cnt]] = x
                    else:
                        x_res[self.residual[cnt]] += x
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())

            elif isinstance(layer, nn.AvgPool2d):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
                    self.pre_acts.append(x.detach().clone())

            elif isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                if self.save:
                    self.acts.append(indices)
                    self.pre_acts.append(indices)

            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())

        x = torch.flatten(x)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                cnt += 1
                x = layer(x)
                if self.save and layer != self.fc_layers[-1]:
                    self.pre_acts.append(x.detach().clone())
            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                if cnt in x_res:
                    x = x + x_res[cnt]
                    del x_res[cnt]
                if cnt in self.residual:
                    if self.residual[cnt] not in x_res:
                        x_res[self.residual[cnt]] = x
                    else:
                        x_res[self.residual[cnt]] += x
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
        return x

    def get_biases(self) -> list[torch.Tensor]:
        b: list[torch.Tensor] = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                b.append(m.bias.data)
        return b
    
    def get_activation_fn(self):
        act_name = self.activation.lower()
        activation_fn_map = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leakyrelu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
            "sigmoid": nn.Sigmoid(),
        }
        if act_name not in activation_fn_map.keys():
            raise ValueError(f"Unknown activation function name : {act_name}")
        return activation_fn_map[act_name]

    def init(self) -> None:
        def init_func(m) -> None:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.randn(m.num_features)
                m.running_var = torch.rand(m.num_features)
            #    m.reset_parameters()
        self.apply(init_func)

    @staticmethod
    def round_up(n: float) -> int:
        # TODO: There are inconsistencies in the output shape of AvgPool2d.
        # Will need to look into it in the future
        return math.floor(n)
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
