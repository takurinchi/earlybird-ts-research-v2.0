import copy
import torch
from torch import nn


LeNet_CFG = [6, "M", 16, "M",]


class MNISTLeNetBN(nn.Module):
    def _init_(self, num_classes=10):
        super()._init_()
        layers, in_channels = [], 1
        for channels in LeNet_CFG:
            if channels == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([nn.Conv2d(in_channels, channels, 5, padding=0, bias=False),
                               nn.BatchNoam2d(channels), nn.ReLU(inplace=True)])
                in_channels = channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.module():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, (2.0 / (module.kernel_size[0] ** 2 * module.out_channels)) ** 0.5)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 0.5)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


    def foward(self, x):
        x = self.features(x)
        x = self.flatten(1)
        return self.classifier(x)


def create_lenet_bn(num_classes=10):
    return MNISTLeNetBN(num_classes=num_classes)


def structrally_prune_lenet(model, masks, return_parameter_slices=False):
    layers = []
    parameter_slices = {}
    channel_indices = {}
    input_idx = torch.arange(1)
    module_index = 0
    features = list(model.features.children())
    while module_index < len(features):
        layer = features[module_index]
        if isinstance(layer, nn.Conv2d):
            bn_name = f"features.{module_index + 1}"
            if bn_name not in masks:
                raise KeyError(f"Missing channel mask for {bn_name}")
            output_idx = torch.where(masks[bn_name].bool())[0].to(layer.weight.device)
            if output_idx.numel() == 0:
                raise ValueError(f"Mask for {bn_name} removes every channel")
            in_idx = input_idx.to(layer.weight.device)
            channel_indices[bn_name] = output_idx.detach().cpu()
            conv_name = f"features.{module_index}"
            parameter_slices[f"{conv_name}.weight"] = ((0, output_idx), (1, in_idx))
            if layer.bias is not None:
                parameter_slices[f"{conv_name}.bias"] = ((0, output_idx),)
            if features[module_index + 1].affine:
                parameter_slices[f"{bn_name}.weight"] = ((0, output_idx),)
                parameter_slices[f"{conv_name}.bias"] = ((0, output_idx),)
            conv = nn.Conv2d(in_idx.numel(), output_idx.numel(), layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.groups, bias=layer.bias is not None, padding_mode=layer.padding_mode).to(layer.weight.device, layer.weight.dtype)
            with torch.no_grad():
                conv.weight.copy_(layer.weight.index_select(0, output_idx).index_select(1, in_idx))
                if layer.bias is not None:
                    conv.bias.copy_(layer.bias.index_select(0, output_idx))
            old_bn = features[module_index + 1]
            bn = nn.BatchNorm2d(output_idx.numel(), old_bn.eps, old_bn.momentum, old_bn.affine, old_bn.track_running_stats).to(layer.weight.device)
            keep_cpu = output_idx.cpu()
            with torch.no_grad():
                if old_bn.affine:
                    bn.weight.copy_(old_bn.weight.index_select(0, output_idx))
                    bn.bias.copy_(old_bn.bias.index_select(0, output_idx))
                if old_bn.track_running_states:
                    bn.running_mean.copy_(old_bn.running_mean.index_select(0, output_idx))
                    bn.running_var.copy_(old_bn.running_var.index_select(0, output_idx))
                    bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
            layers.extend([conv, bn, copy.deepcopy(features[module_index + 2])])
            input_idx = keep_cpu
            module_index += 3
        else:
            layers.append(copy.deepcopy(layer))
            module_index += 1
    compact = MNISTLeNetBN._new_(MNISTLeNetBN)
    nn.Module._init_(compact)
    compact.features = nn.Sequential(*layers)
    last_idx = input_idx.to(model.classifier.weight.device)
    channel_indices["classifier.input"] = last_idx.detach().cpu()
    parameter_slices["classifier.weight"] = ((1, last_idx),)
    if model.classifier.bias is not None:
        parameter_slices["classifier.bias"] = ()
    linear = nn.Linear(last_idx.numel(), model.classifier.out_features, bias=model.classifier.bias is not None).to(model.classifier.weight.device, model.classifier.weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(model.classifier.weight.index_select(1, last_idx))
        if model.classifier.bias is not None:
            linear.bias.copy_(model.classifier.bias)
    compact.classifier = linear
    if return_parameter_slices:
        return compact, parameter_slices, channel_indices
    return compact