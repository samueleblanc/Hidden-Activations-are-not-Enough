import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


class ResNet(nn.Module):
    """
    Implementation of ResNet18.
    This class has a forward method that can be used for training or to 
    save activations and preactivations to later compute the matrix.

    Args:
        input_shape (tuple[int, int, int]): The shape of the input image.
        num_classes (int): The number of classes in the dataset.
        pretrained (bool): Whether to use pretrained weights.
        max_pool (bool): Whether to use max pooling.
        save (bool): Whether to save activations and preactivations.
    """
    def __init__(
            self, 
            input_shape:tuple[int,int,int], 
            num_classes:int, 
            pretrained:bool = True, 
            max_pool:bool = True, 
            save:bool = False
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        c,h,w = input_shape
        self.num_classes = num_classes
        # Dictionary to store the residual connections {start_index: (end_index, downsample_layers)}
        self.residual = {}
        self.bias = True
        self.save = save

        # Initialize the model
        self.weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet18(weights=self.weights, progress=False)
        self.model.eval()
        self.model.to(self.device)
        self.matrix_input_dim = c*w*h + 1

        regular_input = h >= 224
        if regular_input:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225]
                )
            ])

        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        self.conv_layers: list[nn.Module] = []
        self.fc_layers: list[nn.Module] = []
        
        cnt = 0  # Counter for the number of layers
        # Used to identify the start and end of the residual connections
        first_conv = True  # Set to False once the first conv layer is added
        
        # Iterate through the model layers and add them to the conv_layers and fc_layers lists
        for module in self.model.children():
            if isinstance(module, nn.MaxPool2d):
                if max_pool:  # TODO: Currently doesn't work with max pooling (if pretrained is True)
                    if regular_input:
                        self.conv_layers.append(
                            nn.MaxPool2d(
                                kernel_size = module.kernel_size, 
                                padding = module.padding, 
                                stride = module.stride, 
                                return_indices = True
                            )
                        )
                        cnt += 1
                    else:
                        self.conv_layers.append(
                            nn.MaxPool2d(
                                kernel_size = 2, 
                                padding = 0, 
                                stride = 2, 
                                return_indices = True
                            )
                        )
                        cnt += 1
            elif isinstance(module, nn.Sequential):
                for basic_block in module:
                    start = cnt
                    downsample_layers = []
                    for layer in basic_block.children():    
                        if isinstance(layer, nn.Sequential):
                            # Downsample layers are stored in a list 
                            # and added to the self.residual dictionary
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
                if num_classes != 1000:  # Can't use original module if that's the case
                    self.fc_layers.append(
                        nn.Linear(
                            in_features = module.in_features, 
                            out_features = num_classes, 
                            bias = True
                        )
                    )
                else:
                    self.fc_layers.append(module)
                cnt += 1
            elif isinstance(module, nn.Conv2d):
                if first_conv:
                    if c == 3:  # Must have 3 channels to use original module
                        self.conv_layers.append(module)
                    else:
                        self.conv_layers.append(
                            nn.Conv2d(
                                in_channels = c, 
                                out_channels = 64, 
                                kernel_size = 3, 
                                stride = 1, 
                                padding = 1,
                                bias = False
                            )
                        )
                    first_conv = False
                    cnt += 1
            else:
                self.conv_layers.append(module)
                cnt += 1

    def forward(self, x: torch.Tensor, rep:bool=False) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            rep (bool): Whether to save the activations and preactivations.
        Returns:
            torch.Tensor: The output tensor.
        """
        if not rep:
            # Regular forward pass
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

        # Forward pass for matrix computation
        # Save activations and preactivations
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
    
    def get_activation_fn(self) -> nn.ReLU:
        """
        Returns the activation function used in the model.

        Returns:
            nn.ReLU: ReLU activation function
        """
        return nn.ReLU()
