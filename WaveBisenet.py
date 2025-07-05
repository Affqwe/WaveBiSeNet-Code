import torch
from torch import nn
from blocks.build_contextpath import build_contextpath
import warnings
import time
from dysample import DySample_UP
warnings.filterwarnings(action='ignore')
from HWD import HWD
from WTConv import WTConv2d
torch.cuda.empty_cache()
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))
class WHD_WTFConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.MAXPOOL = HWD(in_channels, out_channels)
        self.Conv = WTConv2d(out_channels, out_channels)
        self.Conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)  # 加入1×1卷积
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, input):
        x = self.Conv(self.MAXPOOL(input))
        x = self.Conv1x1(x)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = WHD_WTFConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = WHD_WTFConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = WHD_WTFConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        #x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        # resnet18  1024 = 256(from spatial path) + 256(from context path) + 512(from context path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.maxpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class WaveBisenet(torch.nn.Module):
    def __init__(self, num_classes, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()
        self.up =  DySample_UP(in_channels=64,scale=8,style='pl')#最后8倍后上采样
        self.up_cx1 = DySample_UP(in_channels=256,scale=2,style='pl')
        self.up_cx2 = DySample_UP(in_channels=512, scale=4, style='pl')
        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module  for resnet 101

        if context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(64, 1024)
        else:
            print('Error: unspport context_path network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)            #32

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = self.up_cx1(cx1)   #上采用直接输出
        cx2 = self.up_cx2(cx2)  #上采用直接输出
        cx = torch.cat((cx1, cx2), dim=1)

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = self.up(result)
        result = self.conv(result)


        return result


import os


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available. Please ensure a GPU device is accessible.")
    device = torch.device('cuda')
    model = BiSeNet( 2,'resnet18').to(device)  # 假设输出类别为2
    print(f"Model is on: {device}")

    # 1. 测试模型参数量
    for name,moudle in model.named_modules():
        print(name)
    print("\n--- Model Summary ---")
    model.eval()

    # 创建一个输入张量，Batch Size = 4，输入大小为 (3, 512, 512)
    input_tensor = torch.randn(4, 3, 512, 512).to(device)

    # 测试运行速度
    start_time = time.time()
      # 禁用梯度计算
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()

    print("\n--- Inference Time ---")
    print(f"Time taken for a single forward pass: {end_time - start_time:.6f} seconds")

    # 3. 测试张量输出大小
    print("\n--- Output Tensor Size ---")
    print(f"Output shape: {output.shape}")