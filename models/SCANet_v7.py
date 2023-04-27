import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models



class SCANet(nn.Module):
    def __init__(self, load_weights = False):
        super(SCANet, self).__init__()

        self.front_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,'M', 512, 512, 512, 512]
        self.front1 = make_layers(self.front_feat)      #VGG 16
        self.front2 = make_layers(self.front_feat)      #VGG 16

        self.sca1 = SCA(512)
        self.sca2 = SCA(512)
        self.sca3 = SCA(512)


        self.low_den = Decoder(1024, 1024)
        self.low_conv = nn.Conv2d(512, 128, 1)

        self.middle_den_1 = Decoder(1024, 1024)
        self.middle_den_2 = Decoder(512, 512)
        self.mid_conv = nn.Conv2d(256, 128, 1)

        self.high_den_1 = Decoder(1024, 1024)
        self.high_den_2 = Decoder(512, 512)
        self.high_den_3 = Decoder(256, 256)

        self.reg = nn.Conv2d(128, 1, 1)

        #消融
        #self.reg2 = nn.Conv2d(1024,1,1)


        if not load_weights:
            mod = models.vgg19(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.front1.state_dict().items())):
                list(self.front1.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                list(self.front2.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x1 = x[0]
        x2 = x[1] #测试SCANet网络 需要


        #front
        x1 = self.front1(x1)  #torch.Size([1, 512, 32, 32])
        x2 = self.front2(x2)  #torch.Size([1, 512, 32, 32])


        # middle part - SCA
        o1_first, o2_first, conc1 = self.sca1(x1, x2)                #torch.Size([1, 512, 60, 90])
        o1_second, o2_second, conc2 = self.sca2(o1_first, o2_first)  #torch.Size([1, 512, 60, 90])
        o1_thir, o2_thir, conc3 = self.sca3(o1_second, o2_second)    #torch.Size([1, 512, 60, 90])



        #Grid unit
        x_fore = torch.cat((x1,x2), dim=1)
        o_low_1 = self.low_den(x_fore, conc1)
        o_low = self.low_conv(o_low_1)

        o_middle_1 = self.middle_den_1(conc1, conc2)
        o_middle_2 = self.middle_den_2(o_low_1, o_middle_1)
        o_middle = self.mid_conv(o_middle_2)

        o_high_1 = self.high_den_1(conc2, conc3)
        o_high_2 = self.high_den_2(o_middle_1, o_high_1)
        o_high_3 = self.high_den_3(o_middle_2, o_high_2)


        o = o_low + o_middle + o_high_3

        #back-end
        o = self.reg(o)
        o = F.interpolate(o, scale_factor=2, mode='bilinear')

        return torch.abs(o)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)





class SCA(nn.Module):
    def __init__(self, channel):
        super(SCA, self).__init__()
        self.fra_SA = SpatialAttention_v2(channel)
        self.flow_SA = SpatialAttention_v2(channel)

        self.cross_fra_flow = ChannelAttention(channel*2)

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

        x1_next = x1_next_1 + x1_next_2 + x1
        x2_next = x2_next_1 + x2_next_2 + x2

        x_conc = torch.cat((x1_next,x2_next),dim=1)

        return x1_next, x2_next, x_conc




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

class Decoder(nn.Module):
    def __init__(self, channel1 = 3, channel2 =3):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels= channel1, out_channels= channel1 //2, kernel_size=1)
        self.conv2 = nn.ConvTranspose2d(in_channels= channel2, out_channels=channel2 //2,  kernel_size=3, padding=1)

        channel = channel1 // 2 + channel2 //2
        self.conv3 = nn.Conv2d(in_channels= channel, out_channels= channel //2, kernel_size=1)



    def forward(self, fore_x, curr_x):
        x1 = self.conv1(fore_x)
        x2 = self.conv2(curr_x)

        x_conc = torch.cat((x1, x2),dim=1)

        o = self.conv3(x_conc)


        return o




if __name__ == "__main__":
    #测试
    x1 = torch.rand((1, 3, 256, 256))  #RGB
    x2 = torch.rand((1, 3, 256, 256))  #T
    x = [x1,x2]
    #print(x.shape)
    #x = x.to(device)
    # 瀹氫箟model
    model = SCANet()
    o1 = model(x)
    print(o1.shape)

