import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')


class GCMNet(nn.Module):
    """
    Description
        Full framework for maxillary sinusitis classification project submitted to KCR 2019
        designed for input image size of 256
    """
    def __init__(self, channels, num_classes):
        super(GCMNet, self).__init__()

        # 1 - Feature Extractor
        self.res1 = nn.Sequential(ResidualBlock(channels, 64, False), ResidualBlock(64, 128, True))  # 128,128,128
        self.res2 = nn.Sequential(ResidualBlock(128, 128, False), ResidualBlock(128, 160, True))  # 64,64,160
        self.res3 = nn.Sequential(ResidualBlock(160, 160, False), ResidualBlock(160, 192, True))  # 32,32,192
        self.res4 = nn.Sequential(ResidualBlock(192, 192, False), ResidualBlock(192, 224, True))  # 16,16,224
        self.res5 = nn.Sequential(ResidualBlock(224, 224, False), ResidualBlock(224, 256, True))  # 8,8,256

        # 2 - GCM
        self.gcm = GeometricCorrelationMap()  # 8,8,64

        # 3 - Binary Classifier
        self.cls1 = nn.Sequential(ResidualBlock(64, 64, False), ResidualBlock(64, 128, True))  # 4,4,128
        self.cls2 = nn.Sequential(ResidualBlock(128, 128, False), ResidualBlock(128, 256, True))  # 2,2,256
        self.cls3 = nn.Sequential(ResidualBlock(256, 256, False), ResidualBlock(256, 512, True))  # 1,1,512
        self.logit = nn.Linear(512, num_classes)  # binary classification for both sinuses

    def forward(self, src, dst):
        src_map = self.res5(self.res4(self.res3(self.res2(self.res1(src)))))
        dst_map = self.res5(self.res4(self.res3(self.res2(self.res1(dst)))))
        gcm = self.gcm(src_map, dst_map)
        logit = self.cls3(self.cls2(self.cls1(gcm)))
        logit = self.logit(logit.view(-1, 512))

        return logit


class GeometricCorrelationMap(nn.Module):
    """
    Description
        Generate geometric correlation map which calculates similarity score between two extracted features
        From 'Longitudinal Change Detection on Chest X-rays using Geometric Correlation Maps (2019, MICCAI)'
    Args
        lt_map, rt_map : feature maps which dimensions are (i, j, c) and (x, y, c) each
    Returns
        gcm : feature map with (x, y, i * j) dimension
    """
    def __init__(self):
        super(GeometricCorrelationMap, self).__init__()
        # self.zero = torch.zeros([1]).to(torch.device("cuda:0"))
        self.zero = torch.zeros([1])
        self.gcm = None

    def forward(self, src_map, dst_map):
        src_dim = list(src_map.size())
        src_c, src_h, src_w = src_dim[1], src_dim[2], src_dim[3]

        src_flat = torch.reshape(src_map, (-1, src_c, src_h * src_w))
        src_norm = torch.unsqueeze(torch.norm(src_flat, dim=1), 1)
        dst_norm = torch.unsqueeze(torch.norm(dst_map, dim=1), 1)

        # vector norm
        src = torch.div(src_flat, src_norm)
        dst = torch.div(dst_map, dst_norm)

        for i in range(src_h * src_w):
            src_element = torch.unsqueeze(src[:, :, i:i + 1], -1)  # get a single column
            # print('src element : ', src_element.size())
            gcm_element = torch.unsqueeze(torch.sum(src_element * dst, dim=1), 1)  # calculate correlation score
            gcm_element = torch.max(gcm_element, self.zero)  # zero-out negative values
            if i == 0:
                self.gcm = gcm_element
            else:
                self.gcm = torch.cat((self.gcm, gcm_element), 1)

        return self.gcm


class ResidualBlock(nn.Module):
    """
    Description
        Basic residual block from 'Deep Residual Learning for Image Recognition (2015)'
        Used full pre-activation from 'Identity Mappings in Deep Residual Networks (2016)'
    """
    def __init__(self, in_dim, out_dim, downsample, bias=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(in_dim)

        if self.downsample:
            self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0, bias=bias)
            self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=bias)
        else:
            self.conv_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias)
            self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x)))
        y = self.conv2(F.relu(self.bn2(y)))
        x_proj = self.conv_proj(x)

        return x_proj + y


# model test
# src = torch.randn((32, 1, 8, 8), requires_grad=True)
# dst = torch.randn((32, 1, 8, 8), requires_grad=True)
# model = GeometricCorrelationMap()
# feed = model(src, dst)
# print(feed.size())
#
# src2 = torch.randn((8, 1, 256, 256), requires_grad=True)
# dst2 = torch.randn((8, 1, 256, 256), requires_grad=True)
# model2 = GCMNet(1, 4)
# feed2 = model2(src2, dst2)
# print(feed2.size())
