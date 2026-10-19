import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, vgg11, vgg13, efficientnet_b0, efficientnet_b1, googlenet, inception_v3, shufflenet_v2_x1_0, vit_b_16, efficientnet_b2, efficientnet_b3,  efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
import math

from .SIN_QD.SIN_model import load_model


def get_model(params: Dict) -> nn.Module:
    """
    Factory function to get the model based on the model name.
    Args:
        params (Dict): Dictionary containing model configuration and hyperparameters.
    Returns:
        nn.Module: The model instance.
    """

    img_size = -1
    in_channels = 200  # Default value for raw data management, 41 for IB features, toImage overrides it for image models

    for transformation in params.get("transformations", []):
        if transformation["name"] == "ToImage":
            img_size = transformation["img_size"]
            in_channels = len(transformation["img_channels"])
        if transformation["name"] == "LoopToIBFeatures":
            img_size = 1  # Default image size
            in_channels = 41

    if img_size == -1:
        img_size = 1  # Default image size
        in_channels = 200  # 41

    class_labels = params.get("class_labels")
    num_classes = len(class_labels) if class_labels is not None else 3

    model_name = params["model"]
    weights_enum = 'DEFAULT' if 'pretrained' in model_name else None
    nn_model: Optional[nn.Module] = None

    def freeze_extractor(model):
        """Freezes all parameters except the final fully connected layer (Linear Probing)."""
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'fc'):  # ResNets, Inception, GoogLeNet
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):  # VGG, EfficientNet
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'heads'):  # Vision Transformers
            for param in model.heads.parameters():
                param.requires_grad = True

    def load_qd_checkpoint(arch_fn, pt_filename):
        """Loads a specific QD pre-trained checkpoint and handles layer replacement."""
        # Checkpoints written by SIN_QD/quick_draw.py
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SIN_QD", "result")
        full_path = os.path.join(base_path, pt_filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Model file '{full_path}' does not exist.")

        device = torch.device("cuda" if params.get("gpu", False) else "cpu")

        model = arch_fn(weights=weights_enum)

        # Replace last layer to match the checkpoint's original training classes (QuickDraw 10 classes)
        qd_temp_labels = ['apple', 'boomerang', 'carrot', 'cookie',
                          'hot air balloon', 'mushroom', 'onion', 'pear', 'sun', 'wheel']
        replace_last_fc_layer(nn_model=model, class_labels=qd_temp_labels)

        model.load_state_dict(torch.load(full_path, map_location=device))

        # Replace last layer again for the actual target task
        replace_last_fc_layer_complex(
            nn_model=model, class_labels=params["class_labels"])

        return model

    if model_name == 'baseline':
        # Simple CNN baseline: useful for sanity checks and low-resource testing.
        nn_model = SimpleCNN(num_classes=num_classes, img_size=img_size)

    elif model_name == 'fc_nn':
        # Fully Connected Network: treats input as flat vector, good for non-spatial data.
        nn_model = FullyConnectedNet(
            in_channels=in_channels, num_classes=num_classes, img_size=img_size)

    elif model_name == 'cnn_scales':
        # CNN with scales: simple convolutional network with addtitonal Zrs scale information for images plotted in relative scales.
        nn_model = SimpleCNNWithScales(
            num_classes=num_classes, img_size=img_size, extra_features=4)

    elif model_name == 'time_cnn':
        # 1D CNN for time-series data handling sequential dependencies, preserves temporal information.
        nn_model = TimeAxisCNN(num_axes=4, seq_len=50, num_classes=num_classes)

    elif model_name == 'parametric_time_cnn':
        nn_model = ParametricTimeAxisCNN(
            num_axes=4, seq_len=50, num_classes=num_classes,
            filters_per_axis=params.get("filters_per_axis", 16),
            mixing_filters=params.get("mixing_filters", [64, 128]),
            kernel_size=params.get("kernel_size", 5),
            dropout_rate=params.get("dropout_rate", 0.5),
            fc_units=params.get("fc_units", 128))

    elif model_name == 'flexicnn':
        # Custom flexible CNN architecture allowing depth/width configuration.
        nn_model = FlexiConvNet(
            img_size=img_size, input_channels=in_channels, num_classes=num_classes,
            fe_layers=params.get("fe_layers", 4),
            fc_layers=params.get("fc_layers", 4),
            base_channels=params.get("base_channels", 2),
        )

    elif model_name == 'linearcnn':
        # Custom flexible CNN architecture allowing depth/width configuration.
        nn_model = LinearConvNet(
            img_size=img_size, input_channels=in_channels, num_classes=num_classes,
            fe_layers=params.get("fe_layers", 4),
            fc_layers=params.get("fc_layers", 4),
            growth_factor=params.get("growth_factor", 1.5),
            bottleneck_channels=params.get("bottleneck_channels", 16)
        )
    elif model_name == 'depth_scales':
        # Custom flexible CNN architecture allowing depth/width configuration.
        nn_model = DepthCnnScales(
            img_size=img_size, input_channels=in_channels, num_classes=num_classes,
            fe_layers=params.get("fe_layers", 7),
            fc_layers=params.get("fc_layers", 4),
            growth_factor=params.get("growth_factor", 1.5),
            bottleneck_channels=params.get("bottleneck_channels", 16),
            extra_features=6
        )

    elif model_name == 'depthwise_cnn':
        # Depthwise Separable CNN with Squeeze-Excite blocks: efficient parameter usage.
        nn_model = DepthSepSqueezeExciteNet(
            img_size=img_size, input_channels=in_channels, num_classes=num_classes,
            total_layers=params.get("total_layers", 18),
            feature_split_ratio=params.get("feature_split_ratio", 0.67),
            base_channels_per_input=params.get("base_channels", 2),
        )

    elif 'modified_resnet' in model_name:
        # Determine the base architecture from params
        if 'resnet50' in model_name:
            arch = resnet50
        elif 'resnet34' in model_name:
            arch = resnet34
        else:
            arch = resnet18

        # Instantiate the model (with or without pre-trained weights)
        nn_model = arch(weights=weights_enum)

        # Patch the first convolution layer to accept your specific in_channels
        nn_model = modify_resnet_input_channels(
            nn_model, in_channels=in_channels)

        # Replace the final fully connected layer for your target classes
        replace_last_fc_layer(nn_model=nn_model, class_labels=class_labels)

    # Standard ResNet Architectures (ImageNet weights)
    elif any(x in model_name for x in ['resnet50', 'resnet34', 'resnet18']) and 'QD' not in model_name and 'SIN' not in model_name:
        # Standard ResNet backbones (50, 34, 18). Good general purpose image extractors.
        if 'resnet50' in model_name:
            arch = resnet50
        elif 'resnet34' in model_name:
            arch = resnet34
        elif 'resnet18' in model_name:
            arch = resnet18

        nn_model = arch(weights=weights_enum)
        replace_last_fc_layer(nn_model=nn_model, class_labels=class_labels)

    # Other Standard Torchvision Models
    elif any(x in model_name for x in ['vgg', 'efficientnet', 'googlenet', 'vit', 'shufflenet', 'inception']):
        if 'vgg11' in model_name:
            nn_model = vgg11(weights=weights_enum)
        elif 'vgg13' in model_name:
            nn_model = vgg13(weights=weights_enum)
        elif 'efficientnet_b0' in model_name:
            nn_model = efficientnet_b0(weights=weights_enum)
        elif 'efficientnet_b1' in model_name:
            nn_model = efficientnet_b1(weights=weights_enum)
        elif 'efficientnet_b2' in model_name:
            nn_model = efficientnet_b2(weights=weights_enum)
        elif 'efficientnet_b3' in model_name:
            nn_model = efficientnet_b3(weights=weights_enum)
        elif 'efficientnet_b4' in model_name:
            nn_model = efficientnet_b4(weights=weights_enum)
        elif 'efficientnet_b5' in model_name:
            nn_model = efficientnet_b5(weights=weights_enum)
        elif 'efficientnet_b6' in model_name:
            nn_model = efficientnet_b6(weights=weights_enum)
        elif 'efficientnet_b7' in model_name:
            nn_model = efficientnet_b7(weights=weights_enum)
        elif 'googlenet' in model_name:
            nn_model = googlenet(weights=weights_enum, aux_logits=False)
        elif 'vit-b-16' in model_name:
            nn_model = vit_b_16(weights=weights_enum)
        elif 'shufflenet' in model_name:
            nn_model = shufflenet_v2_x1_0(weights=weights_enum)
        elif 'inception' in model_name:
            nn_model = inception_v3(weights=weights_enum, aux_logits=False)

        # Generic last layer replacement logic
        if hasattr(
                nn_model, 'classifier') and isinstance(
                nn_model.classifier, nn.Sequential):  # type: ignore
            # For VGG-like structures where classifier is a container
            layer = nn_model.classifier[-1]  # type: ignore
            in_ftrs = layer.in_features if hasattr(
                layer, 'in_features') else layer.out_features
            # type: ignore
            # type: ignore
            nn_model.classifier[-1] = nn.Linear(int(in_ftrs), num_classes)
        elif hasattr(nn_model, 'classifier') and not isinstance(nn_model.classifier, nn.Sequential):  # type: ignore
            # EfficientNet style
            in_ftrs = nn_model.classifier[-1].in_features  # type: ignore
            nn_model.classifier[-1] = nn.Linear(in_ftrs, num_classes)  # type: ignore
        elif hasattr(nn_model, 'fc'):
            # ResNet/Inception style
            nn_model.fc = nn.Linear(nn_model.fc.in_features,
                                    num_classes)  # type: ignore
        elif hasattr(nn_model, 'heads'):
            # ViT style
            nn_model.heads.head = nn.Linear(
                nn_model.heads.head.in_features, num_classes)  # type: ignore

    # Stylized-ImageNet (SIN) Pre-trained Models
    elif 'SIN' in model_name:
        # Models pre-trained on Stylized ImageNet (SIN) to improve shape bias over texture bias.
        source_name = 'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN' if 'IMGNET' in model_name else 'resnet50_trained_on_SIN'
        sin_model: torch.nn.DataParallel = load_model(source_name)
        print(f"Loaded SIN pre-trained model: {source_name}")
        print(f"model type: {type(nn_model)}")
        nn_model = sin_model.module  # unwrap DataParallel
        replace_last_fc_layer(
            nn_model=nn_model, class_labels=class_labels)  # type: ignore

        if 'frozen' in model_name:
            freeze_extractor(nn_model)

    # QuickDraw (QD) Custom Pre-trained Models
    elif 'QD' in model_name:
        params['qd'] = True
        # Models pre-trained on custom QuickDraw data, and the trained model might be pretrained on imagenet.
        if 'resnet50' in model_name and 'pretrained' in model_name:
            fname = "QD_resnet50_pretrained.pt"
            arch_fn = resnet50
        elif 'resnet50' in model_name and 'pretrained' not in model_name:
            fname = "QD_resnet50.pt"
            arch_fn = resnet50
        elif 'resnet18' in model_name and 'pretrained' in model_name:
            fname = "QD_resnet18_pretrained.pt"
            arch_fn = resnet18
        elif 'resnet18' in model_name and 'pretrained' not in model_name:
            fname = "QD_resnet18.pt"
            arch_fn = resnet18
        else:
            raise ValueError(f"Unknown QD model configuration: {model_name}")

        nn_model = load_qd_checkpoint(arch_fn, fname)

        if 'frozen' in model_name:
            freeze_extractor(nn_model)

    else:
        raise ValueError(f"Model {model_name} is not recognized.")

    assert isinstance(nn_model, nn.Module), "Model creation failed, nn_model is None"

    return nn_model


def replace_last_fc_layer(nn_model: torch.nn.Module,
                          class_labels: Optional[list[str]] = None):
    num_ftrs: int = nn_model.fc.in_features  # type: ignore
    nn_model.fc = nn.Linear(num_ftrs, len(class_labels)
                            if class_labels is not None else 3)


def replace_last_fc_layer_complex(
        nn_model: torch.nn.Module, class_labels: Optional[list[str]] = None):
    num_ftrs: int = nn_model.fc.in_features  # type: ignore
    num_classes = len(class_labels) if class_labels is not None else 3
    hidden1 = max(256, int(num_ftrs) // 2)
    hidden2 = 128

    nn_model.fc = nn.Sequential(
        nn.Linear(int(num_ftrs), hidden1),
        nn.BatchNorm1d(hidden1),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(hidden1, hidden2),
        nn.BatchNorm1d(hidden2),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(hidden2, num_classes)
    )


class SimpleCNN(nn.Module):
    def __init__(
            self, in_channels: int = 1, num_classes: int = 4, dropout_prob: float = 0.5,
            img_size: int = 128):
        """
        Define the layers of the convolutional neural network with batch normalization,
        dropout, and a simple attention map mechanism.

        Parameters:
            in_channels: int
                The number of channels in the input image.
            num_classes: int
                The number of classes to predict.
            dropout_prob: float
                Dropout probability.
        """
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(4*img_size*img_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor after passing through the network's fully connected layer, shape (batch_size, num_classes).
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class FullyConnectedNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 3,
                 dropout_prob: float = 0.5,
                 img_size: int = 128,
                 hidden_layer_sizes: Tuple[int] = (40, 20, 10),  # type: ignore
                 alpha: float = 0.01
                 ):
        """
        Fully connected network with optional L2 regularization helper.

        Parameters:
            in_channels: int
                Number of channels in the input image.
            num_classes: int
                Number of classes to predict.
            dropout_prob: float
                Dropout probability (applied before the final output layer).
            img_size: int
                Height/width of the (square) input image.
            hidden_layer_sizes: Tuple[int]
                Hidden layer sizes, e.g. (40, 20, 10).
            alpha: float
                Strength of the L2 regularization term. The L2 term returned by
                l2_regularization(...) is already divided by the supplied sample
                size (batch_size).
        """
        super(FullyConnectedNet, self).__init__()

        self.input_features = in_channels * img_size * img_size
        self.alpha = float(alpha)

        hidden_layers = []
        in_dim = self.input_features

        for h_dim in hidden_layer_sizes:
            hidden_layers.append(nn.Linear(in_dim, h_dim))
            hidden_layers.append(nn.BatchNorm1d(h_dim))
            hidden_layers.append(nn.ReLU(inplace=True))
            in_dim = h_dim

        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc_out = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, num_classes).
        """
        x = x.reshape(x.shape[0], -1)
        x = self.hidden_layers(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

    def l2_regularization(self, batch_size: int) -> torch.Tensor:
        """
        Compute L2 regularization term to add to the loss.

        The returned tensor equals:
            alpha * sum(params^2) / batch_size

        Bias terms and 1D parameters are excluded (only weight matrices / tensors with
        ndim > 1 are included).

        Args:
            batch_size: int
                The sample size used to divide the L2 term.

        Returns:
            torch.Tensor: Scalar tensor containing the L2 regularization term.
        """
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        reg = torch.tensor(0.0, device=device)
        for name, p in self.named_parameters():
            if p.requires_grad and p.ndim > 1:
                reg = reg + p.pow(2).sum()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0 for l2_regularization.")
        return (self.alpha * reg) / float(batch_size)


class SimpleCNNWithScales(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 3,
                 dropout_prob: float = 0.5,
                 img_size: int = 128,
                 extra_features: int = 4,
                 ):
        """
        Define the layers of a convolutional neural network that also accepts
        extra tabular features.

        Parameters:
            in_channels: int
                The number of channels in the input image.
            num_classes: int
                The number of classes to predict.
            dropout_prob: float
                Dropout probability.
            img_size: int
                The height/width of the (square) input image.
            extra_features: int
                The number of extra features to be concatenated before the
                final fully connected layer.
        """
        super(SimpleCNNWithScales, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        flattened_img_features = 4 * img_size * img_size
        combined_features = flattened_img_features + extra_features
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(combined_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward accepts a single input. It supports:
          - a tuple/list (x_img, x_extra) (keeps previous behavior), or
          - a 4D tensor (B, C_total, H, W) where the first C_img channels are the image
            and the remaining channels are "extra" feature maps (these are spatially pooled
            to produce per-sample extra features).

        The image channels count is inferred from self.conv1.in_channels.
        """
        if isinstance(x, (list, tuple)):
            x_img, x_extra = x

            # Normalize different possible shapes for x_extra into (B, C_extra)
            if isinstance(x_extra, torch.Tensor):
                # Case: (B, C_extra, H, W)
                if x_extra.dim() == 4:
                    x_extra = x_extra.mean(dim=(2, 3))
                # Case: (C_extra, H, W) -> pool to (C_extra,) then expand to batch
                elif x_extra.dim() == 3:
                    pooled = x_extra.mean(dim=(1, 2))  # (C_extra,)
                    x_extra = pooled.to(device=x_img.device, dtype=x_img.dtype).unsqueeze(
                        0).expand(x_img.size(0), -1)
                # Case: (B, C_extra) already good
                elif x_extra.dim() == 2:
                    if x_extra.size(0) != x_img.size(0):
                        if x_extra.size(0) == 1:
                            x_extra = x_extra.expand(x_img.size(
                                0), -1).to(device=x_img.device, dtype=x_img.dtype)
                        else:
                            raise ValueError(
                                "Batch size mismatch between x_img and x_extra.")
                # Case: single vector (C_extra,) -> expand to batch
                elif x_extra.dim() == 1:
                    x_extra = x_extra.to(device=x_img.device, dtype=x_img.dtype).unsqueeze(
                        0).expand(x_img.size(0), -1)
                else:
                    raise ValueError("Unsupported x_extra tensor shape.")
            else:
                raise ValueError(
                    "Expected x_extra to be a torch.Tensor when passing a tuple/list.")
        else:
            if x.dim() != 4:
                raise ValueError(
                    "Expected either (img, extra) tuple or a 4D tensor (B, C, H, W).")
            in_ch = self.conv1.in_channels
            if x.shape[1] < in_ch:
                raise ValueError(
                    f"Input has {x.shape[1]} channels but model expects at least {in_ch}.")
            x_img = x[:, :in_ch, :, :]
            if x.shape[1] == in_ch:
                # no extra channels provided -> create empty extra tensor
                x_extra = torch.empty(
                    (x.size(0), 0), device=x.device, dtype=x.dtype)
            else:
                extra_maps = x[:, in_ch:, :, :]  # (B, C_extra, H, W)
                x_extra = extra_maps.mean(dim=(2, 3))  # (B, C_extra)

        x = F.relu(self.bn1(self.conv1(x_img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        combined = torch.cat((x, x_extra), dim=1)
        combined = self.dropout(combined)
        combined = self.fc1(combined)
        return combined


class DepthCnnScales(nn.Module):
    def __init__(self,
                 img_size=128,
                 input_channels=1,
                 num_classes=4,
                 fe_layers=7,
                 fc_layers=4,
                 growth_factor=1.5,
                 max_width=64,
                 extra_features=4,
                 bottleneck_channels=16):

        super(DepthCnnScales, self).__init__()

        self.num_feature_layers = fe_layers
        self.num_classifier_layers = fc_layers
        self.extra_features = extra_features
        self.input_channels = input_channels
        feature_layers = []
        current_channels = input_channels
        next_channels = max(current_channels + 1,
                            math.floor(current_channels * growth_factor))

        for i in range(self.num_feature_layers):
            feature_layers.extend([
                nn.Conv2d(current_channels, current_channels, kernel_size=3,
                          padding=1, groups=current_channels),
                nn.Conv2d(current_channels, next_channels, kernel_size=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU()
            ])
            if i % 2 == 0:
                feature_layers.append(nn.MaxPool2d(2, 2))

            current_channels = next_channels
            potential_next = math.floor(next_channels * growth_factor)
            next_channels = min(potential_next, max_width)

        self.features = nn.Sequential(*feature_layers)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, bottleneck_channels, kernel_size=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU()
        )

        dummy_input = torch.zeros(1, input_channels, img_size, img_size)
        with torch.no_grad():
            feat_out = self.features(dummy_input)
            bn_out = self.bottleneck(feat_out)

        self.flattened_img_size = bn_out.view(1, -1).size(1)
        total_fc_input = self.flattened_img_size + extra_features

        print(f"Image Feature Map: {bn_out.shape[1:]}")
        print(f"Total Linear Input: {total_fc_input}")

        classifier_layers = []
        layer_sizes = np.linspace(total_fc_input, num_classes, fc_layers + 1, dtype=int)

        for i in range(len(layer_sizes) - 1):
            in_f = int(layer_sizes[i])
            out_f = int(layer_sizes[i+1])

            classifier_layers.append(nn.Linear(in_f, out_f))
            if i < len(layer_sizes) - 2:
                classifier_layers.append(nn.ReLU())
                classifier_layers.append(nn.Dropout(0.3))

        self.classifier = nn.Sequential(*classifier_layers)
        self._init_extra_feature_weights(5)

    def _init_extra_feature_weights(self, scale):
        """
        Manually scales the weights of the extra features in the first linear layer.
        """
        first_fc = self.classifier[0]

        with torch.no_grad():
            if self.extra_features > 0:
                first_fc.weight[:, -self.extra_features:] *= scale  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x_img, x_extra = x
            if x_extra.dim() == 4:
                x_extra = x_extra.mean(dim=(2, 3))
            elif x_extra.dim() == 1:
                x_extra = x_extra.unsqueeze(0).expand(x_img.size(0), -1)
        else:
            # Assume x is (B, C_total, H, W)
            in_ch = self.input_channels
            x_img = x[:, :in_ch, :, :]
            if x.shape[1] > in_ch:
                x_extra = x[:, in_ch:, :, :].mean(dim=(2, 3))
            else:
                x_extra = torch.empty((x.size(0), 0), device=x.device, dtype=x.dtype)
        x = self.features(x_img)
        x = self.bottleneck(x)
        x = torch.flatten(x, 1)
        combined = torch.cat((x, x_extra), dim=1)
        return self.classifier(combined)


class TimeAxisCNN(nn.Module):
    def __init__(self, num_axes=4, seq_len=50, num_classes=2):
        """
        A more robust CNN that preserves temporal information and has
        higher capacity to break the 75% accuracy ceiling.
        """
        super(TimeAxisCNN, self).__init__()
        filters_per_axis = 16
        self.independent_conv = nn.Conv1d(
            in_channels=num_axes,
            out_channels=num_axes * filters_per_axis,
            kernel_size=5,
            padding=2,
            groups=num_axes
        )
        self.bn1 = nn.BatchNorm1d(num_axes * filters_per_axis)
        self.mixing_conv1 = nn.Conv1d(
            in_channels=num_axes * filters_per_axis,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.mixing_conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.flatten_size = 128 * 12
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.independent_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.mixing_conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.mixing_conv2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ParametricTimeAxisCNN(nn.Module):
    def __init__(self,
                 num_axes=4,
                 seq_len=50,
                 num_classes=2,
                 filters_per_axis=16,
                 mixing_filters=[64, 128],
                 kernel_size=5,
                 dropout_rate=0.5,
                 fc_units=128):
        super(ParametricTimeAxisCNN, self).__init__()

        self.independent_conv = nn.Conv1d(
            in_channels=num_axes,
            out_channels=num_axes * filters_per_axis,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=num_axes
        )
        self.bn1 = nn.BatchNorm1d(num_axes * filters_per_axis)

        self.mixing_layers = nn.ModuleList()
        self.mixing_bns = nn.ModuleList()

        in_ch = num_axes * filters_per_axis
        for out_ch in mixing_filters:
            self.mixing_layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
            )
            self.mixing_bns.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)

        self.flatten_size = self._get_conv_output_shape(num_axes, seq_len)

        self.fc1 = nn.Linear(self.flatten_size, fc_units)
        self.fc2 = nn.Linear(fc_units, num_classes)

    def _get_conv_output_shape(self, num_axes, seq_len):
        """Helper to calculate the size of the flattened features automatically."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_axes, seq_len)
            x = self.pool(F.relu(self.bn1(self.independent_conv(dummy_input))))
            for i, (conv, bn) in enumerate(zip(self.mixing_layers, self.mixing_bns)):
                x = conv(x)
                x = bn(x)
                x = F.relu(x)
                if i == 0:
                    x = self.pool(x)
            return x.numel()

    def forward(self, x):
        x = F.relu(self.bn1(self.independent_conv(x)))
        x = self.pool(x)

        for i, (conv, bn) in enumerate(zip(self.mixing_layers, self.mixing_bns)):
            x = F.relu(bn(conv(x)))
            if i == 0:
                x = self.pool(x)

        x = x.flatten(1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FlexiConvNet(nn.Module):
    """
   A convolutional neural network with a flexible, user-defined architecture.

    This model allows users to specify the exact number of layers dedicated to 
    feature extraction (convolutional) and classification (fully connected). 
    It automatically calculates the size of the flattened feature map after the 
    convolutional layers to ensure a seamless connection to the fully connected layers.
    """

    def __init__(self,
                 img_size=64,
                 input_channels=3,
                 num_classes=3,
                 fe_layers=8,
                 fc_layers=8,
                 base_channels=16):
        """
        Initializes the FlexiConvNet architecture.

        Args:
            img_size (int): The height and width of the input images (assumed to be square).
            input_channels (int): The number of channels in the input images. Default is 3.
            num_classes (int): The number of target output classes for classification.
            fe_layers (int): The number of convolutional layers for feature extraction.
            fc_layers (int): The number of parametric fully connected layers for classification. 
            base_channels (int): The number of output channels for the first convolutional layer.
                                 Note: This number doubles after every single feature layer.
        """

        super(FlexiConvNet, self).__init__()

        # Ensure at least one feature and one classifier layer is created
        self.num_feature_layers = fe_layers if fe_layers >= 1 else 1
        self.num_classifier_layers = fc_layers if fc_layers >= 1 else 1

        feature_layers = []
        current_channels = input_channels
        next_channels = base_channels

        # Feature extraction layers with increasing channels and periodic pooling
        for i in range(self.num_feature_layers):
            # Append Conv2d -> BatchNorm -> ReLU block
            feature_layers.append(
                nn.Conv2d(current_channels, next_channels,
                          kernel_size=3, padding=1)
            )
            feature_layers.append(nn.BatchNorm2d(next_channels))
            feature_layers.append(nn.ReLU())

            # Append MaxPool2d every 2 layers to reduce spatial dimensions
            if i % 2 == 0:
                feature_layers.append(nn.MaxPool2d(2, 2))

            current_channels = next_channels
            next_channels = next_channels * 2

        self.features = nn.Sequential(*feature_layers)

        # Automatically calculate the flattened size by passing a dummy tensor through the feature extractor
        dummy_input = torch.zeros(1, input_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)

        print(f"Calculated Flattened Size: {flattened_size}")

        # Classifier layers with a linear reduction in size from the flattened feature map to the number of classes
        classifier_layers = []
        max_width = 2048
        current_neurons = flattened_size
        target_start_width = min(flattened_size, max_width)

        # Step size for linear reduction across classifier layers
        if self.num_classifier_layers > 1:
            step = (target_start_width -
                    num_classes) // self.num_classifier_layers
        else:
            step = 0

        for i in range(self.num_classifier_layers):
            if i == 0:
                next_neurons = target_start_width
            elif i == self.num_classifier_layers - 1:
                next_neurons = num_classes
            else:
                next_neurons = max(current_neurons - step, num_classes + 1)

            # Add activation and dropout for all hidden layers (skip for the final output layer)
            classifier_layers.append(nn.Linear(current_neurons, next_neurons))
            if i < self.num_classifier_layers - 1:
                classifier_layers.append(nn.ReLU())
                classifier_layers.append(nn.Dropout(0.3))

            current_neurons = next_neurons

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network: input goes through feature extractor, is flattened, and then goes through the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        """
        Calculates the total number of trainable parameters in the model.
        Returns a dictionary with 'trainable', 'non_trainable', and 'total'.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non-trainable": total_params - trainable_params
        }


class LinearConvNet(nn.Module):
    """
    A convolutional neural network with flexible architecture, utilizing a linear 
    reduction strategy for the fully connected layers.

    Distinct from FlexiConvNet, this model employs a 1x1 bottleneck convolution 
    prior to flattening and reduces the size of the fully connected layers 
    linearly using numpy's linspace.
    """

    def __init__(self,
                 img_size=64,
                 input_channels=3,
                 num_classes=10,
                 fe_layers=8,
                 fc_layers=8,
                 growth_factor=1.5,
                 max_width: int = 512,
                 bottleneck_channels: int = 16):
        """
        Initializes the LinearConvNet architecture.

        Args:
            img_size (int): The height and width of the input images (assumed to be square).
            input_channels (int): The number of channels in the input images.
            num_classes (int): The number of target output classes for classification.
            fe_layers (int): The number of convolutional layers for feature extraction.
            fc_layers (int): The desired number of fully connected layers (creates fc_layers + 1 layers). The +1 is because the final layer is the output layer with num_classes neurons.
            growth_factor (float): The multiplier used to grow the number of channels per layer.
            max_width (int): The maximum allowed number of neurons in the first linear layer.
            bottleneck_channels (int): The number of channels outputted by the 1x1 bottleneck layer.
        """

        super(LinearConvNet, self).__init__()

        # Ensure at least one feature and one classifier layer is specified
        self.num_feature_layers = fe_layers if fe_layers >= 1 else 1
        self.num_classifier_layers = fc_layers if fc_layers >= 1 else 1

        feature_layers = []
        current_channels = input_channels
        next_channels = math.floor(input_channels * growth_factor)

        # Build convolutional feature extraction layers with growth in channels and periodic pooling
        for i in range(self.num_feature_layers):
            # Append Conv2d -> BatchNorm -> ReLU block
            feature_layers.append(
                nn.Conv2d(current_channels, next_channels,
                          kernel_size=3, padding=1)
            )
            feature_layers.append(nn.BatchNorm2d(next_channels))
            feature_layers.append(nn.ReLU())

            # Append MaxPool2d every 2 layers to reduce spatial dimensions
            if i % 2 == 0:
                feature_layers.append(nn.MaxPool2d(2, 2))

            current_channels = next_channels
            next_channels = math.floor(
                next_channels * growth_factor) if next_channels * 2 <= 64 else 64

        self.features = nn.Sequential(*feature_layers)

        # Add a 1x1 convolutional bottleneck layer to reduce the number of channels before flattening
        bottleneck_layers = []
        bottleneck_layers.append(
            nn.Conv2d(current_channels, bottleneck_channels,
                      kernel_size=1, padding=0)
        )
        bottleneck_layers.append(nn.BatchNorm2d(bottleneck_channels))
        bottleneck_layers.append(nn.ReLU())

        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Automatically calculate the flattened size by passing a dummy tensor through the feature extractor and bottleneck
        dummy_input = torch.zeros(1, input_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
            dummy_output = self.bottleneck(dummy_output)
        flattened_size = dummy_output.view(1, -1).size(1)

        print(f"Calculated Flattened Size: {flattened_size}")

        classifier_layers = []
        current_neurons = flattened_size
        target_start_width = min(flattened_size, max_width)

        # Calculate a linear step for reduction
        target_layer_sizes = np.linspace(
            target_start_width, num_classes, self.num_classifier_layers + 1, dtype=int)

        for i in range(self.num_classifier_layers+1):
            next_neurons = int(target_layer_sizes[i])

            classifier_layers.append(nn.Linear(current_neurons, next_neurons))

            # Don't add ReLU/Dropout to the very last output layer
            if i < self.num_classifier_layers:
                classifier_layers.append(nn.ReLU())
                classifier_layers.append(nn.Dropout(0.3))

            current_neurons = next_neurons

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network: input goes through feature extractor, then bottleneck, is flattened, and then goes through the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.bottleneck(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        """
        Calculates the total number of trainable parameters in the model.
        Returns a dictionary with 'trainable', 'non_trainable', and 'total'.
        """
        total_params = sum(p.numel() for p in self.parameters())

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non-trainable": total_params - trainable_params
        }


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.

    This block improves the representational power of the network by explicitly 
    modeling the interdependencies between the channels of its convolutional features.
    It "squeezes" spatial information into a channel descriptor and "excites" the 
    channels by scaling them based on their learned importance.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels (int): The number of input channels.
            reduction (int): The reduction ratio used to create the bottleneck.
                             Higher values compress the representation more.
        """
        super(SEBlock, self).__init__()

        # Calculate bottleneck channels, enforcing a minimum of 4 to prevent
        # dropping to 0 or too few channels on highly reduced layers.
        squeezed_channels = max(channels // reduction, 4)

        # Squeeze operation: Global Average Pooling (H, W -> 1, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation operation: Two fully connected layers acting as a bottleneck,
        # outputting a scaling factor (between 0 and 1) for each channel.
        self.excitation = nn.Sequential(
            nn.Linear(channels, squeezed_channels),
            nn.ReLU(),
            nn.Linear(squeezed_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        # Squeeze: pool spatial dimensions, then reshape to (batch, channels)
        y = self.squeeze(x).view(b, c)

        # Excite: pass through MLP to get channel weights, reshape for broadcasting
        y = self.excitation(y).view(b, c, 1, 1)

        # Scale the original input feature map by the learned channel weights
        return x * y


class DepthwiseChannelBlock(nn.Module):
    """
    A block consisting of a spatial depthwise convolution, followed by a grouped 
    pointwise expansion, and an optional Squeeze-and-Excitation module.
    """

    def __init__(
        self,
        in_channels: int,
        channels_per_input: int,
        stride: int = 1,
        use_se: bool = True,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            channels_per_input (int): The expansion factor per individual input channel.
            stride (int): Stride for the depthwise convolution (controls spatial downsampling).
            use_se (bool): Whether to append a Squeeze-and-Excitation block at the end.
        """
        super().__init__()

        # Total output channels is a strict multiple of the input channels
        out_channels = in_channels * channels_per_input

        # 1. Depthwise Convolution: Filters spatial dimensions independently per channel
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,  # 1 filter per input channel
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # 2. Pointwise Expansion: 1x1 convolution
        # information across different original input channels. It independently
        # expands each input channel into `channels_per_input` separate channels.
        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # 3. Optional Channel Attention
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        return x


class DepthSepSqueezeExciteNet(nn.Module):
    """
    A dynamic Convolutional Neural Network that automatically partitions a specified 
    total number of layers into a feature extraction phase and a classification phase.

    It relies heavily on grouped depthwise convolutions and applies Squeeze-and-Excitation 
    sparsely (on every other layer) to maintain parameter efficiency.
    """

    def __init__(
        self,
        img_size: int = 224,
        input_channels: int = 3,
        num_classes: int = 10,
        total_layers: int = 12,
        feature_split_ratio: float = 0.7,
        base_channels_per_input: int = 2,
        max_channels: int = 90,
        dropout: float = 0.1,
    ):
        """
        Args:
            img_size (int): Height and width of input images.
            input_channels (int): Initial input channels (e.g., 3 for RGB).
            num_classes (int): Number of target classes.
            total_layers (int): The total combined layers for feature + classifier blocks.
            feature_split_ratio (float): Percentage of total_layers devoted to feature extraction (0.0 to 1.0).
            base_channels_per_input (int): The initial channel expansion multiplier.
            max_channels (int): A hard cap to prevent massive channel explosion in deeper layers.
            dropout (float): Dropout probability for the classifier block.
        """
        super().__init__()

        # Apportion layers based on the requested split ratio
        self.num_feature_layers = max(
            1, int(math.floor(total_layers * feature_split_ratio))
        )
        self.num_classifier_layers = max(
            1, total_layers - self.num_feature_layers
        )

        feature_layers = []
        in_ch = input_channels
        channels_per_input = base_channels_per_input

        # Define a specific rhythm for spatial downsampling (stride=2 at layers 0, 3, 6...)
        downsample_points = {i for i in range(0, self.num_feature_layers, 3)}

        # Feature extraction phase: Stacked DepthwiseChannelBlocks with increasing channels and periodic downsampling
        for i in range(self.num_feature_layers):
            stride = 2 if i in downsample_points else 1

            feature_layers.append(
                DepthwiseChannelBlock(
                    in_channels=in_ch,
                    channels_per_input=channels_per_input,
                    stride=stride,
                    # Apply SE blocks sparsely (only on odd indices)
                    use_se=(i % 2 == 1),
                )
            )

            # Update channel counts for the next block
            in_ch = in_ch * channels_per_input

            # Calculate the next expansion multiplier, capped by max_channels
            channels_per_input = min(
                channels_per_input * 2,
                max_channels // max(in_ch, 1),
            )
            # Ensure the expansion multiplier never drops below 1
            channels_per_input = max(channels_per_input, 1)

        self.features = nn.Sequential(*feature_layers)

        # Automatically determine the flattened feature map size via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, img_size, img_size)
            dummy_out = self.features(dummy)
            self.flattened_feature_dim = dummy_out.numel() // dummy_out.shape[0]

        # Classifier phase: A series of fully connected layers with a linear reduction strategy, culminating in the final output layer matching num_classes
        classifier_layers = []
        current_dim = self.flattened_feature_dim

        # Target width for the classifier's internal layers before hitting num_classes
        target_dim = max(num_classes * 8, 128)

        # Calculate a linear reduction step size
        if self.num_classifier_layers > 1:
            step = max((current_dim - target_dim) // self.num_classifier_layers, 1)
        else:
            step = 0

        # Build hidden classification layers
        for _ in range(self.num_classifier_layers - 1):
            next_dim = max(current_dim - step, target_dim)

            classifier_layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])

            current_dim = next_dim

        # Final output layer mapping to the exact number of classes
        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        # Output architecture summary to console during initialization
        self._print_model_summary()

    def _print_model_summary(self):
        print("DepthSepSqueezeExciteNet (no GAP) constructed")
        print(f"Feature layers: {self.num_feature_layers}")
        print(f"Classifier layers: {self.num_classifier_layers}")
        print(f"Flattened feature dim: {self.flattened_feature_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.
        Note: This architecture flattens the feature map directly instead of 
        using Global Average Pooling (GAP) prior to classification.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def modify_resnet_input_channels(model: nn.Module, in_channels: int):
    """
    Modifies the first conv layer of a ResNet to accept arbitrary channels.
    If pre-trained weights are present, it copies the average weights across 
    the new channels (for BW/1-channel) or initializes them.
    """
    if in_channels == 3:
        return model

    old_conv = model.conv1

    # Create the new convolution layer matching your input channels
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # Smart weight initialization if you are using pre-trained weights
    with torch.no_grad():
        if in_channels == 1:
            # For 1-channel (grayscale), sum/average the RGB weights
            new_conv.weight.copy_(old_conv.weight.sum(dim=1, keepdim=True))
        else:
            # For multi-channel (e.g., 4 channels), reuse RGB for the first 3
            # and initialize the remaining channels
            if in_channels == 2:
                # For 2-channel inputs, copy the first two RGB channel weights
                new_conv.weight[:, :2].copy_(old_conv.weight[:, :2])
            else:
                new_conv.weight[:, :3].copy_(old_conv.weight)
            if in_channels > 3:
                # Initialize extra channels with the average of the RGB weights
                avg_weight = old_conv.weight.mean(dim=1, keepdim=True)
                for i in range(3, in_channels):
                    new_conv.weight[:, i:i+1].copy_(avg_weight)

    model.conv1 = new_conv
    return model
