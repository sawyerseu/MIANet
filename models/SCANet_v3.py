import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class SCANet(nn.Module):
    def __init__(self, pretrained=False, ratio=1):
        super(SCANet,self).__init__()

        self.seen = 0

        self.block1 = Block([int(64 * ratio), int(64 * ratio), 'M'], first_block=True)
        self.block2 = Block([int(128 * ratio), int(128 * ratio), 'M'], in_channels=int(64 * ratio))
        self.block3 = Block([int(256 * ratio), int(256 * ratio), int(256 * ratio), int(256 * ratio), 'M'], in_channels=int(128 * ratio))
        self.block4 = Block([int(512 * ratio), int(512 * ratio), int(512 * ratio), int(512 * ratio), 'M'], in_channels=int(256 * ratio))

        self.block5 = Block([int(512 * ratio), int(512 * ratio), int(512 * ratio), int(512 * ratio)], in_channels=int(512 * ratio))

        self.conv1 = nn.Conv2d(128, 512, 3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(256, 512, 3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(512, 512, 3, stride=1, padding=1, dilation=1)
        self.conv4 = nn.Conv2d(1024, 512, 3, stride=1, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(1024, 512, 3, stride=1, padding=1, dilation=1)

        self.comc = nn.Conv2d(512*5, 512, 1)

        self.dila1 = nn.Conv2d(512, 256, 3, stride=1, padding=1, dilation=1)
        self.dila2 = nn.Conv2d(512, 256, 3, stride=1, padding=2, dilation=2)
        self.dila3 = nn.Conv2d(512, 256, 3, stride=1, padding=3, dilation=3)


        self.backend_feat = [int(3*256), int(512 * ratio), int(512 * ratio), int(256 * ratio), int(128 * ratio),
                             64]
        self.backend = make_layers(self.backend_feat, in_channels=int(256*3), d_rate=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if pretrained:
            self._initialize_weights(mode='normal')
        else:
            self._initialize_weights(mode='kaiming')

    def forward(self, RGBT):
        x1 = RGBT[0]
        x2 = RGBT[1]

        o1_first, o2_first = self.block1(x1, x2)
        o1_second, o2_second = self.block2(o1_first, o2_first)
        o1_thir, o2_thir = self.block3(o1_second, o2_second)
        o1_four, o2_four = self.block4(o1_thir, o2_thir)
        o1_five, o2_five = self.block5(o1_four, o2_four)

        con_fir = torch.cat((o1_first, o2_first), dim=1)     #torch.Size([1, 1024, 60, 90])
        con_secon = torch.cat((o1_second, o2_second), dim=1)   #torch.Size([1, 1024, 60, 90])
        con_thir = torch.cat((o1_thir, o1_thir),dim=1)         #torch.Size([1, 1024, 60, 90])
        con_four = torch.cat((o1_four, o2_four), dim=1)  # torch.Size([1, 1024, 60, 90])
        con_five = torch.cat((o1_five, o2_five), dim=1)  # torch.Size([1, 1024, 60, 90])

        # back part
        back_feat1 = self.conv1(con_fir)  # torch.Size([1, 512, 128, 128])
        back_feat2 = self.conv2(con_secon)  # torch.Size([1, 512, 64, 64])
        back_feat3 = self.conv3(con_thir)  # torch.Size([1, 512, 32, 32])
        back_feat4 = self.conv4(con_four)  # torch.Size([1, 512, 32, 32])
        back_feat5 = self.conv4(con_five)  # torch.Size([1, 512, 32, 32])

        # optional operation,
        # back_add1 = (back_feat1) + (back_feat2) + (back_feat3)  #1*512*30*30
        # back_add1 = (back_feat1) * (back_feat2) * (back_feat3)  #1*512*30*30
        # back_add1 = torch.cat((back_feat1,back_feat2,back_feat3), dim=1)  #需要改后面的接口

        back_feat1 = F.interpolate(back_feat1, scale_factor=1/4, mode='bilinear')
        back_feat2 = F.interpolate(back_feat2, scale_factor=1/2, mode='bilinear')
        back_feat3 = F.interpolate(back_feat3, scale_factor=1, mode='bilinear')
        back_feat4 = F.interpolate(back_feat4, scale_factor=2, mode='bilinear')
        back_feat5 = F.interpolate(back_feat5, scale_factor=2, mode='bilinear')

        # 1-Addition
        back_add1 = 0.00005 * (back_feat1) + 0.0005 * (back_feat2) + 0.005 * (back_feat3) + 0.05*(back_feat4) + 1*(back_feat5)# 1*512*30*30

        # 2-multipication
        #back_add1 =  (back_feat1) * (back_feat2)  * (back_feat3) * (back_feat4) # 1*512*30*30

        # 3-conc
        #back_add1 = torch.cat((back_feat1,back_feat2,back_feat3,back_feat4),dim=1)
        #back_add1 = self.comc(back_add1)

        back_dila1 = self.dila1(back_add1)  # torch.Size([1, 256, 58, 88])
        back_dila2 = self.dila2(back_add1)  # torch.Size([1, 256, 58, 88])
        back_dila3 = self.dila3(back_add1)  # torch.Size([1, 256, 58, 88])

        back_add2 = torch.cat((back_dila1, back_dila2, back_dila3), 1) #1 1024 128 128

        #back_feat = torch.cat((o1_five,o2_five),dim=1)
        #back_feat = F.interpolate(back_feat, scale_factor=2, mode='bilinear')
        fusion_feature = self.backend(back_add2)
        map = self.output_layer(fusion_feature)  # torch.Size([1, 256, 60, 90])

        return torch.abs(map)

    def _initialize_weights(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == 'normal':
                    nn.init.normal_(m.weight, std=0.01)
                elif mode == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Block(nn.Module):
    def __init__(self, cfg, in_channels = 3, first_block = False, dilation_rate =1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate

        if first_block is True:
            rgb_in_channels = 3
            t_in_channels = 3
        else:
            rgb_in_channels = in_channels
            t_in_channels = in_channels

        self.rgb_conv = make_layers(cfg, in_channels=rgb_in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=t_in_channels, d_rate=self.d_rate)

        channel = cfg[0]
        self.fra_SA = SpatialAttention_v2(channel)
        self.flow_SA = SpatialAttention_v2(channel)

        self.cross_fra_flow = ChannelAttention(channel*2)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, RGB, T):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)

        new_RGB, new_T = self.sca(RGB, T)

        return new_RGB, new_T

    def sca(self, RGB, T):

        x1 = RGB
        x2 = T

        x1_sa = self.fra_SA(x1)
        x2_sa = self.flow_SA(x2)

        x_con = torch.cat((x1, x2), dim=1)
        x_cross_att = self.cross_fra_flow(x_con)
        x_cross_att = self.mlp(x_cross_att)

        x1_next_1 = x1 * x1_sa
        x2_next_1 = x2 * x2_sa
        temp_att = x_cross_att * x1_sa

        x1_next_2 = x1 * x_cross_att
        x2_next_2 = x2 * x_cross_att

        x1_next = x1 * x1_sa * x_cross_att + x1
        x2_next = x2 * x2_sa * x_cross_att + x2

        return x1_next, x2_next




#channel-wise attention
class ChannelAttention(nn.Module):
    def __init__(self, channel=3, reduction=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


# 没有用
class SCA(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=3):
        super(SCA, self).__init__()
        self.fra_SA = SpatialAttention(kernel_size)
        self.flow_SA = SpatialAttention(kernel_size)

        self.cross_fra_flow = ChannelAttention(channel*2, reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel*2, channel, 1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1_sa = self.fra_SA(x1)
        x2_sa = self.flow_SA(x2)

        x_con = torch.cat((x1,x2), dim=1)
        x_cross_att = self.cross_fra_flow(x_con)
        x_cross_att = self.mlp(x_cross_att)

        x1_next_1 = x1 * x1_sa
        x2_next_1 = x2 * x2_sa
        temp_att =  x_cross_att * x1_sa

        x1_next_2 = x1 * x_cross_att
        x2_next_2 = x2 * x_cross_att

        x1_next = x1_next_1 * x1_next_2 + x1
        x2_next = x2_next_1 * x2_next_2 + x2

        return x1_next, x2_next


#spatial attention v2
class SpatialAttention_v2(nn.Module):
    def __init__(self, channel = 3):
        super(SpatialAttention_v2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels= channel, out_channels= channel, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels= channel, out_channels= channel, kernel_size=1),
            nn.Conv2d(in_channels= channel, out_channels= channel, kernel_size=3, padding=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, padding=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=7, padding=3)
        )
        self.out = nn.Conv2d(in_channels= 4* channel, out_channels= 1, kernel_size= 1)

        self.sigmoid = nn.Sigmoid()

    def forward (self, x):

        out1 = self.conv1(x)
        out2 = self.conv1(x)
        out3 = self.conv1(x)
        out4 = self.conv1(x)
        out_conc = torch.cat((out1,out2,out3,out4),dim=1)
        out_temp = self.out(out_conc)
        out_att = self.sigmoid(out_temp)

        return out_att


#spatial attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)  #note 这里512测试网络用
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    #测试
    x1 = torch.rand((1, 3, 256, 256)).to(device)     #RGB
    x2 = torch.rand((1, 3, 256, 256)).to(device) # T
    x = [x1, x2]
    model = SCANet().to(device)
    o1 = model(x)
    print(o1.shape)