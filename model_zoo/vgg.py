"""
    Implementation of VGG11
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg11, vgg11_bn, VGG11_Weights, VGG11_BN_Weights


class VGG(nn.Module):

    def __init__(self, input_shape: tuple[int,int,int], num_classes: int, pretrained:bool=True, 
                 max_pool:bool=True, batch_norm:bool=False, save:bool=False) -> None:
        super().__init__()
        self.input_shape = input_shape
        c,h,w = input_shape
        self.num_classes = num_classes
        self.bias = True
        self.batch_norm = batch_norm
        self.save = save

        if pretrained:
            if batch_norm:
                self.weights = VGG11_BN_Weights.DEFAULT
            else:
                self.weights = VGG11_Weights.DEFAULT
        else:
            self.weights = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = vgg11_bn(weights=self.weights, progress=False) if batch_norm else vgg11(weights=self.weights, progress=False)
        self.model.eval()
        self.model.to(self.device)
        self.matrix_input_dim = c*w*h + 1

        regular_input = h >= 224
        if regular_input:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        self.conv_layers = []
        self.fc_layers = []
        
        first_conv = True
        fc = 0
        for module in self.model.children():
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.MaxPool2d):
                        if max_pool:
                            if regular_input:
                                self.conv_layers.append(nn.MaxPool2d(kernel_size=layer.kernel_size, padding=layer.padding, stride=layer.stride, return_indices=True))
                            else:
                                self.conv_layers.append(nn.MaxPool2d(kernel_size=3, padding=0, stride=1, return_indices=True))

                    elif isinstance(layer, nn.Linear):
                        if fc == 2 and num_classes != 1000:
                            self.fc_layers.append(nn.Linear(layer.in_features, num_classes, bias=True))
                        else:
                            self.fc_layers.append(layer)
                        fc += 1

                    elif isinstance(layer, nn.Conv2d):
                        if first_conv:
                            if c == 3:
                                self.conv_layers.append(layer)
                            else:
                                self.conv_layers.append(nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False))
                            first_conv = False
                        else:
                            self.conv_layers.append(layer)

                    else:
                        if fc == 0:  # Not yet in the fully connected part of the NN
                            self.conv_layers.append(layer)
                        else:
                            self.fc_layers.append(layer)
            
            elif isinstance(module, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                self.conv_layers.append(module)

    def forward(self, x: torch.Tensor, rep=False) -> torch.Tensor:
        if not rep:
            if len(x.shape) == 3: x = x.unsqueeze(0)
            for layer in self.conv_layers:
                if isinstance(layer, nn.MaxPool2d):
                    x, _ = layer(x)
                else:
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

        x = x.unsqueeze(0)
        x = self.conv_layers[0](x)  # Conv

        if self.save:
            self.pre_acts.append(x.detach().clone())
            if self.batch_norm:
                self.acts.append(x.detach().clone())
        if self.batch_norm:
            x = self.conv_layers[1](x)  # BN
            if self.save:
                self.pre_acts.append(x.detach().clone())
            x = self.conv_layers[2](x)  # Activation
            if self.save:
                self.acts.append(x.detach().clone())
        else:
            x = self.conv_layers[1](x)  # Activation
            if self.save:
                self.acts.append(x.detach().clone())

        for i in range(3 if self.batch_norm else 2, len(self.conv_layers)):
            layer = self.conv_layers[i]

            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())
                    if self.batch_norm:
                        self.acts.append(x.detach().clone())
            elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                x = layer(x)
                if self.save:
                    self.pre_acts.append(x.detach().clone())
                    self.acts.append(x.detach().clone())
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
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
                x = layer(x)
                if self.save and layer != self.fc_layers[-1]:
                    self.pre_acts.append(x.detach().clone())
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                if self.save:
                    self.acts.append(x.detach().clone())
        return x
    
    def get_activation_fn(self):
        return nn.ReLU()
