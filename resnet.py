import torch
import torch.nn as nn
import functools


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CropLayer(nn.Module):

    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, attention_module=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)

        center_offset_from_origin_border = 0
        ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
        hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)

        self.ver_conv_crop_layer = nn.Identity()
        ver_conv_padding = ver_pad_or_crop
        self.hor_conv_crop_layer = nn.Identity()
        hor_conv_padding = hor_pad_or_crop

        self.ver_conv = nn.Conv2d(inplanes, planes, kernel_size=(3, 1), padding=ver_conv_padding, stride=stride,
                                  bias=False)
        self.hor_conv = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), padding=hor_conv_padding, stride=stride,
                                  bias=False)
        self.ver_bn = nn.BatchNorm2d(planes)
        self.hor_bn = nn.BatchNorm2d(planes)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        if attention_module is not None:
            if type(attention_module) == functools.partial:
                module_name = attention_module.func.get_module_name()
            else:
                module_name = attention_module.get_module_name()

            if module_name == "simam":
                self.conv2 = nn.Sequential(
                    self.conv2,
                    attention_module(planes)
                )
            else:
                self.bn2 = nn.Sequential(
                    self.bn2, 
                    attention_module(planes)
                )
            
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        ver_out1 = self.ver_conv_crop_layer(x)
        ver_out1 = self.ver_conv(ver_out1)
        ver_out1 = self.ver_bn(ver_out1)

        hor_out1 = self.hor_conv_crop_layer(x)
        hor_out1 = self.hor_conv(hor_out1)
        hor_out1 = self.hor_bn(hor_out1)

        out = out + ver_out1 + hor_out1

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)  

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attention_module=None, 
                 deep_stem=False, stem_width=32, avg_down=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = stem_width*2 if deep_stem else 64

        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        if deep_stem:
            self.conv1 = nn.Sequential(
                conv3x3(3, stem_width, stride=2),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, stem_width, stride=1),
                norm_layer(stem_width),
                nn.ReLU(),
                conv3x3(stem_width, self.inplanes, stride=1),
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes if not deep_stem else stem_width*2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], avg_down=avg_down,
                                       attention_module=attention_module)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[0],
                                       attention_module=attention_module)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[1],
                                       attention_module=attention_module)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, avg_down=avg_down,
                                       dilate=replace_stride_with_dilation[2],
                                       attention_module=attention_module)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # print(self.modules)
    
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False, dilate=False, attention_module=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if avg_down and stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, count_include_pad=False, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion, 1),
                    norm_layer(planes * block.expansion)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    norm_layer(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, attention_module))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attention_module=attention_module))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)

    return model


def resnet18(**kwargs):

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], **kwargs)



def resnet34(**kwargs):

    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], **kwargs)
