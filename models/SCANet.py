import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models



class SCANet(nn.Module):
    def __init__(self,channel, load_weights = False):
        super(SCANet, self).__init__()

        self.front_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.front1 = make_layers(self.front_feat)      #VGG 16
        self.front2 = make_layers(self.front_feat)      #VGG 16

        self.sca1 = SCA(512)
        self.sca2 = SCA(512)
        self.sca3 = SCA(512)

        self.conv1 = nn.Conv2d(1024, 512, 3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(1024, 512, 3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(1024, 512, 3, stride=1, padding=1, dilation=1)

        self.dila1 = nn.Conv2d(512, 256, 3, stride=1, padding=1, dilation=1)
        self.dila2 = nn.Conv2d(512, 256, 3, stride=1, padding=2, dilation=2)
        self.dila3 = nn.Conv2d(512, 256, 3, stride=1, padding=3, dilation=3)

        self.den = nn.Sequential(
                                nn.ConvTranspose2d(768, 256, 4, stride=2, padding=1, output_padding=0, bias=True),
                                nn.PReLU(),
                                nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
                                nn.PReLU(),
                                nn.ConvTranspose2d(64, 8, 4, stride=2, padding=1, output_padding=0, bias=True),
                                nn.PReLU(),
                                nn.Conv2d(8, 1, 1)
                                )
        self.den2 = nn.Sequential(
                                nn.Conv2d(768, 256, 3, 1, 1, 1),
                                nn.Conv2d(256, 128, 3, 1, 1, 1),
                                nn.Conv2d(128, 64, 3, 1, 1, 1),
                                nn.Conv2d(64, 1, 3, 1, 1, 1),
        )

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.front1.state_dict().items())):
                list(self.front1.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
                list(self.front2.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self, x):
        x1 = x[0]
        x2 = x[1] #测试SCANet网络 需要

        #x1 = torch.unsqueeze(x1, dim=0)
        #x2 = torch.unsqueeze(x2, dim=0)
        #x1 = x[0]
        #x2 = x[1]

        #front
        x1 = self.front1(x1)  #torch.Size([1, 512, 32, 32])
        x2 = self.front2(x2)  #torch.Size([1, 512, 32, 32])


        # middle part - SCA
        o1_first, o2_first = self.sca1(x1, x2)                #torch.Size([1, 512, 60, 90])
        o1_second, o2_second = self.sca2(o1_first, o2_first)  #torch.Size([1, 512, 60, 90])
        o1_thir, o2_thir = self.sca3(o1_second, o2_second)    #torch.Size([1, 512, 60, 90])

        con_fir = torch.cat((o1_first, o2_first), dim=1)     #torch.Size([1, 1024, 60, 90])
        con_secon = torch.cat((o1_second, o2_second), dim=1)   #torch.Size([1, 1024, 60, 90])
        con_thir = torch.cat((o1_thir, o1_thir),dim=1)         #torch.Size([1, 1024, 60, 90])

        #back part
        back_feat1 = self.conv1(con_fir)    # torch.Size([1, 512, 58, 88])
        back_feat2 = self.conv2(con_secon)  # torch.Size([1, 512, 58, 88])
        back_feat3 = self.conv3(con_thir)   # torch.Size([1, 512, 58, 88])

        #optional operation,
        # back_add1 = (back_feat1) + (back_feat2) + (back_feat3)  #1*512*30*30
        # back_add1 = (back_feat1) * (back_feat2) * (back_feat3)  #1*512*30*30
        #back_add1 = torch.cat((back_feat1,back_feat2,back_feat3), dim=1)  #需要改后面的接口
        back_add1 = 1 / 8 * (back_feat1) + 1 / 16 * (back_feat2) + 1 / 32 * (back_feat3)  # 1*512*30*30

        back_dila1 = self.dila1(back_add1)   #torch.Size([1, 256, 58, 88])
        back_dila2 = self.dila2(back_add1)   #torch.Size([1, 256, 58, 88])
        back_dila3 = self.dila3(back_add1)   #torch.Size([1, 256, 58, 88])

        back_add2 = torch.cat((back_dila1,back_dila2,back_dila3),1)

        back_reg1 = self.den2(back_add2)  #torch.Size([1, 256, 60, 90])

        return torch.abs(back_reg1)


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




#channel-wise attention
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
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




if __name__ == "__main__":
    #测试
    x1 = torch.rand((1, 3, 256, 256))  #RGB
    x2 = torch.rand((1, 3, 256, 256))  #T
    x = torch.cat([x1,x2], dim=0)
    #print(x.shape)
    #x = x.to(device)
    # 瀹氫箟model
    model = SCANet(3)
    o1 = model(x)
    print(o1.shape)

