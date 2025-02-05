"""
    Implementation of ResNet
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


class ResNet(nn.Module):

    def __init__(self, input_shape: tuple[int,int,int], num_classes: int, max_pool:bool=True, save:bool=False) -> None:
        super().__init__()
        self.input_shape = input_shape
        c,h,w = input_shape
        self.num_classes = num_classes
        self.residual = {}
        self.bias = True
        self.save = save

        self.weights = ResNet18_Weights.DEFAULT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet18(weights=self.weights, progress=False)
        self.model.eval()
        self.model.to(self.device)
        self.matrix_input_dim = c*w*h + 1

        regular_input = h >= 224 and num_classes == 1000
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

        self.conv_layers = []
        self.fc_layers = []
        
        cnt = 0
        first_conv = True
        for module in self.model.children():
            if isinstance(module, nn.MaxPool2d):
                if max_pool:  # TODO: Currently doesn't work with max pooling
                    if regular_input:
                        self.conv_layers.append(nn.MaxPool2d(kernel_size=module.kernel_size, padding=module.padding, stride=module.stride, return_indices=True))
                        cnt += 1
                    else:
                        self.conv_layers.append(nn.MaxPool2d(kernel_size=2, padding=0, stride=2, return_indices=True))
                        cnt += 1
            elif isinstance(module, nn.Sequential):
                for basic_block in module:
                    start = cnt
                    downsample_layers = []
                    for layer in basic_block.children():            
                        if isinstance(layer, nn.Sequential):  # Downsample
                            for ds_layer in layer.children():
                                downsample_layers.append(ds_layer)
                        elif isinstance(layer, nn.Linear):
                            self.fc_layers.append(layer)
                            cnt += 1
                        else:
                            self.conv_layers.append(layer)
                            cnt += 1
                    self.residual[start] = (cnt, downsample_layers)
                    self.conv_layers.append(nn.ReLU())
                    cnt += 1
            elif isinstance(module, nn.Linear):
                if regular_input:
                    self.fc_layers.append(module)
                else:
                    self.fc_layers.append(nn.Linear(module.in_features, num_classes, bias=True))
                cnt += 1
            elif isinstance(module, nn.Conv2d):
                if first_conv:
                    if regular_input:
                        self.conv_layers.append(module)
                    else:
                        self.conv_layers.append(nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False))
                    first_conv = False
                    cnt += 1
            else:
                self.conv_layers.append(module)
                cnt += 1

    def forward(self, x: torch.Tensor, rep=False) -> torch.Tensor:
        if not rep:
            if len(x.shape) == 3: x = x.unsqueeze(0)
            cnt = 0
            x_res = {}
            for layer in self.conv_layers:
                if cnt in self.residual: x_res[self.residual[cnt][0]] = (cnt, x)
                if cnt in x_res:
                    pos, downsample = x_res[cnt]
                    for l in self.residual[pos][1]:
                        downsample = l(downsample)
                    x = x + downsample
                    del x_res[cnt]
                cnt += 1
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
        cnt = 0
        x_res = {}

        x = x.unsqueeze(0)
        if cnt in self.residual: x_res[self.residual[cnt][0]] = (cnt, x)
        x = self.conv_layers[0](x)  # Conv
        cnt += 1

        if self.save:
            self.pre_acts.append(x.detach().clone())
            self.acts.append(x.detach().clone())
        x = self.conv_layers[1](x)  # BN
        cnt += 1
        if self.save:
            self.pre_acts.append(x.detach().clone())
        if cnt in x_res:
            pos, downsample = x_res[cnt]
            for l in self.residual[pos][1]:
                downsample = l(downsample)
            x = x + downsample
            del x_res[cnt]
        if cnt in self.residual: x_res[self.residual[cnt][0]] = (cnt, x)
        x = self.conv_layers[2](x)  # Activation
        cnt += 1
        if self.save:
            self.acts.append(x.detach().clone())

        for i in range(3, len(self.conv_layers)):
            layer = self.conv_layers[i]
            if cnt in self.residual: x_res[self.residual[cnt][0]] = (cnt, x)
            if cnt in x_res:
                pos, downsample = x_res[cnt]
                for l in self.residual[pos][1]:
                    downsample = l(downsample)
                x = x + downsample
                del x_res[cnt]
            cnt += 1

            if isinstance(layer, (nn.Conv2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
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


resnet = ResNet((1,64,64), 1000)

