import torch.nn as nn
#import resnet
#import dilated_conv_resnet_all as resnet
import standard_resnet_all as resnet
from torch.nn import functional as F
import torch
import os
import sys

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet.resnet50(pretrained=pretrained)
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
        # 就是layer1,layer2，layer3，layer4
            self.blocks.append(list(self.model.children())[num_this_layer])

    def forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out
class ResNet101(nn.Module):
    def __init__(self, pretrained=True):
        """Declare all needed layers."""
        super(ResNet101, self).__init__()
        self.model = resnet.resnet101(pretrained=pretrained)
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
        # 就是layer1,layer2，layer3，layer4
            self.blocks.append(list(self.model.children())[num_this_layer])

    def forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out

class Classifier(nn.Module):
    def __init__(self, in_features=2048, num_class=20):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_features, num_class)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        return x


 

class ACB(nn.Module):
    def __init__(self, in_channel,out_channel,kernel_size=3):
        super(ACB, self).__init__()

        self.conv_k1 = nn.Conv2d(in_channel, out_channel, kernel_size=(kernel_size,1), padding =((kernel_size-1)//2,0))
        self.conv_1k = nn.Conv2d(in_channel, out_channel, kernel_size=(1,kernel_size), padding =(0,(kernel_size-1)//2))
        self.conv3x3 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.conv_k1(x)
        x_1 = self.bn(x_1)

        x_2 = self.conv_1k(x)
        x_2 = self.bn(x_2)

        x_3 = self.conv3x3(x)
        x_3 = self.bn(x_3)

        x_end  = x_1 + x_2 + x_3
        return self.relu(x_end)       

class Mask_Classifier(nn.Module):
    def __init__(self, in_features=256, num_class=21):
        super(Mask_Classifier, self).__init__()
        self.mask_conv = nn.Conv2d(in_features, num_class, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.mask_conv(x)
        return x
        
# GAU change ; 
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        
        self.fc1 = nn.Conv2d(channels_high,channels_high//4,1,bias = False)
        self.fc2 = nn.Conv2d(channels_high//4,channels_low,1,bias = False)
        self.sigmoid = nn.Sigmoid()
        if upsample:
            #self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.conv3x3_high = nn.Conv2d(channels_high, channels_low, kernel_size=3, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):
        
        b, c, h, w = fms_high.shape
        b_low,c_low,h_low,w_low = fms_low.shape
        ##############################################################################################
        fms_high_avg = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_max = nn.MaxPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        ##############################################################################################  
        fms_high_gp = fms_high_avg+fms_high_max
        #fms_high_gp = fms_high_avg
        
        
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)
        ##############################################################################################
        #fms_high_gp = self.sigmoid(fms_high_gp)
        ##############################################################################################

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            fms_high = self.conv3x3_high(fms_high)
            fms_high = self.bn_upsample(fms_high)
            fms_high = self.relu(fms_high)
            out = self.relu(
                #self.bn_upsample(self.conv_upsample(fms_high)) + fms_att)
                torch.nn.functional.upsample(input=fms_high,size=(h_low,w_low),scale_factor=None,mode='bilinear',align_corners=True)+fms_att)
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out

class PAN(nn.Module):
    def __init__(self, blocks=[],args=None):
        """
        :param blocks: Blocks of the network with reverse sequential.
        """
        super(PAN, self).__init__()
        channels_blocks = []
        
        
        self.bata_53 = args.bata53
        self.bata_52 = args.bata52
        self.bata_43 = args.bata43
        self.bata_42 = args.bata42
        self.bata_32 = args.bata32
        
        
        for i, block in enumerate(blocks):
            channels_blocks.append(list(list(block.children())[2].children())[4].weight.shape[0])

        self.cab0 = ACB(channels_blocks[0],channels_blocks[0])
        self.cab1 = ACB(channels_blocks[1],channels_blocks[1])
        self.cab2 = ACB(channels_blocks[2],channels_blocks[2])
        self.cab3 = ACB(channels_blocks[3],channels_blocks[3])
        
        self.sa = SpatialAttention(kernel_size = 7)
        
        
        #self.gau_block1 = GAU(channels_blocks[0], channels_blocks[1], upsample=False)
        self.gau_block54 = GAU(channels_blocks[0], channels_blocks[1])
        self.gau_block53 = GAU(channels_blocks[0],channels_blocks[2])
        self.gau_block52 = GAU(channels_blocks[0],channels_blocks[3])
        self.gau_block43 = GAU(channels_blocks[1], channels_blocks[2])
        self.gau_block42 = GAU(channels_blocks[1], channels_blocks[3])
        self.gau_block32 = GAU(channels_blocks[2], channels_blocks[3])
        #self.gau = [self.gau_block1, self.gau_block2, self.gau_block3]
        
        self.conv3x3 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1, bias=False)
        self.bn_high = nn.BatchNorm2d(1024)
        

        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, fms=[]):
        """
        :param fms: Feature maps of forward propagation in the network with reverse sequential. shape:[b, c, h, w]
        :return: fm_high. [b, 256, h, w]
        """
        feature_blocks = []
        
        #res5:fms[0],res4:fms[1],res3:fms[2],res2:fms[3]
        deC5_b,deC5_c,deC5_h,deC5_w=fms[0].shape
        deC4_b,deC4_c,deC4_h,deC4_w=fms[1].shape
        deC3_b,deC3_c,deC3_h,deC3_w=fms[2].shape
        deC2_b,deC2_c,deC2_h,deC2_w=fms[3].shape
        
        fms[0] = self.cab0(fms[0])
        fms[1] = self.cab1(fms[1])
        fms[2] = self.cab2(fms[2])
        fms[3] = self.cab3(fms[3])

        fm_5 = fms[0]
        fm_4 = self.gau_block54(fm_5,fms[1])
        fm_3 = self.bata_53 * self.gau_block53(fm_5,fms[2])+self.bata_43 * self.gau_block43(fm_4,fms[2])
        fm_2 = self.bata_52 * self.gau_block52(fm_5,fms[3])+self.bata_42 * self.gau_block42(fm_4,fms[3]) + self.bata_32 * self.gau_block32(fm_3,fms[3])
        
        
        return fm_2
        

class Mask_Classifier(nn.Module):
    def __init__(self, in_features=256, num_class=21):
        super(Mask_Classifier, self).__init__()
        self.mask_conv = nn.Conv2d(in_features, num_class, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.mask_conv(x)
        return x
        
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)





