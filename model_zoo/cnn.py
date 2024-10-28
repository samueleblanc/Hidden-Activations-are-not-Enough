"""
    2D CNN
"""

import torch
import torch.nn as nn


class CNN_2D(nn.Module):
    def __init__(
        self, input_shape: tuple[int,int,int], num_classes: int, channels:tuple[int]=(8, 16), fc:tuple[int]=(500), kernel_size:tuple[int,int]=(3, 3), 
        bias:bool=False, residual:bool=False, batch_norm:bool=False, dropout:bool=False, num_skipped_layers:bool=3,save:bool=False) -> None:
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        self.channels = channels
        self.num_skipped_layers = num_skipped_layers
        self.save = save
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.kernel = kernel_size[0]
        self.residual = residual
        self.bias = bias
        self.batch_norm = batch_norm
        self.dropout = True #dropout
        self.matrix_input_dim = c*w*h + 1 if bias or batch_norm else c*w*h

        # Conv2d layers are hardcoded for padding=(1, 1) which keeps the shape (channels[0],w,w) after it is applied
        self.conv_layers.append(nn.Conv2d(in_channels=input_shape[0],
                                          out_channels=channels[0] if isinstance(channels, tuple) else channels,
                                          kernel_size=kernel_size,
                                          padding=(1, 1),
                                          bias=False))
        
        if batch_norm: self.conv_layers.append(nn.BatchNorm2d(channels[0] if isinstance(channels, tuple) else channels))

        if isinstance(channels, tuple):
            for i in range(1, len(channels)):
                self.conv_layers.append(nn.ReLU())
                self.conv_layers.append(nn.Conv2d(in_channels=channels[i-1],
                                                out_channels=channels[i],
                                                kernel_size=kernel_size,
                                                padding=(1, 1),
                                                bias=False))
                if batch_norm: self.conv_layers.append(nn.BatchNorm2d(channels[i]))

        if self.dropout: self.conv_layers.append(nn.Dropout(0.25))
        self.conv_layers.append(nn.ReLU())
        # Average pooling is hardcoded for kernel_size=4 and padding=0
        self.conv_layers.append(nn.AvgPool2d(kernel_size=4, padding=0))
        self.fc_layers.append(nn.Linear(in_features=(channels[-1] if isinstance(channels, tuple) else channels) * (input_shape[1]//4) * (input_shape[2]//4),
                                        # This is for no average pooling after the last convolution
                                        #in_features=channels[-1] * input_shape[1] * input_shape[2],
                                        out_features=fc[0] if isinstance(fc, tuple) else fc,
                                        bias=bias))

        if isinstance(fc, tuple):
            for i in range(1, len(fc)):
                self.fc_layers.append(nn.ReLU())
                if dropout: self.fc_layers.append(nn.Dropout(0.5))
                self.fc_layers.append(nn.Linear(in_features=fc[i-1],
                                                out_features=fc[i],
                                                bias=bias))
        self.fc_layers.append(nn.ReLU())
        # if dropout: self.fc_layers.append(nn.Dropout(0.5))
        self.fc_layers.append(nn.Linear(in_features=fc[-1] if isinstance(fc, tuple) else fc,
                                        out_features=num_classes,
                                        bias=bias))

    def forward(self, x: torch.Tensor, rep=False) -> torch.Tensor:
        if not rep:
            for layer in self.conv_layers:
                x = layer(x)
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
            else:
                x = torch.flatten(x)
            for layer in self.fc_layers:
                x = layer(x)
            return x

        self.pre_acts: list[torch.Tensor] = []
        self.acts: list[torch.Tensor] = []

        if rep:  # This is when computing representations
            x = x.unsqueeze(0)
        else:  # This is for validation and test
            x = x.squeeze()

        x = self.conv_layers[0](x)  # Conv

        if self.save:
            self.pre_acts.append(x.detach().clone())
            if self.batch_norm:
                self.acts.append(x.detach().clone())
        if self.batch_norm:
            x = self.conv_layers[1](x)  # BN
            if self.save:
                self.pre_acts.append(x.detach().clone())
            x = self.conv_layers[2](x)  # RELU
            if self.save:
                self.acts.append(x.detach().clone())
        else:
            x = self.conv_layers[1](x)  # RELU
            if self.save:
                self.acts.append(x.detach().clone())

        x_res = x
        cont = 0
        initial = 3 if self.batch_norm else 2

        for i in range(initial, len(self.conv_layers)):
            layer = self.conv_layers[i]
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                cont += 1
                if self.save:
                    self.pre_acts.append(x.detach().clone())

            elif isinstance(layer, (nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.PReLU, nn.Sigmoid)):
                if self.residual:
                    if cont % self.num_skipped_layers == 0:  # Add the saved input after a residual pair
                        x = x + x_res
                        x = layer(x)
                        x_res = x
                    else:
                        x = layer(x)
                else:
                    x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())

            elif isinstance(layer, nn.AvgPool2d):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
                    self.pre_acts.append(x.detach().clone())

            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
                    self.pre_acts.append(x.detach().clone())

        x = torch.flatten(x)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if self.save and layer != self.fc_layers[-1]:
                    self.pre_acts.append(x.detach().clone())
            elif isinstance(layer, nn.ReLU):
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

    def init(self) -> None:
        def init_func(m) -> None:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.randn(m.num_features)
                m.running_var = torch.rand(m.num_features)
            #    m.reset_parameters()
        self.apply(init_func)