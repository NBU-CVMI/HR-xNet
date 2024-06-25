import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.autograd import Function
import torch.nn as nn

from lib.models.channel_shuffle import channel_shuffle
class sign_(nn.Module):

    def __init__(self, *kargs, **kwargs):
        super(sign_, self).__init__(*kargs, **kwargs)
        self.r = sign_f.apply  ### <-----注意此处

    def forward(self, inputs):
        outs = self.r(inputs)
        return outs

class sign_f(Function):
    @staticmethod
    def forward(ctx, inputs):
        output = inputs.new(inputs.size())
        output[inputs >= 100.] = inputs
        output[inputs < 100.] = 0
        ctx.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_output[input_ > 100.] = 1
        grad_output[input_ < 100.] = 0
        return grad_output


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes // 8)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.125, stride=1, padding=1, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)


        self.weight1 = nn.Parameter(torch.randn(K, in_planes, 1, kernel_size, kernel_size), requires_grad=True)


        self.PW = nn.Conv2d(in_planes,out_planes,1,1,0)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight1[i])



    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        res = x
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight1 = self.weight1.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight1).view(-1, 1, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.in_planes, output.size(-2), output.size(-1))

        output = self.PW(output)
        # res connect

        return output


class Dynamic_conv2dMultiKernel(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.125, stride=1, padding=1, dilation=1, groups=1, bias=True, K=2,temperature=40, init_weight=True):
        super(Dynamic_conv2dMultiKernel, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = padding
        self.padding = 1
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K

        self.conv = nn.Conv2d(3,3,3,1,1,4)
        self.attention1 = attention2d(in_planes, ratio, K, temperature)

        self.attention2 = attention2d(int(in_planes), ratio, K, temperature)

        self.attention3 = attention2d(int(in_planes), ratio, K, temperature)

        self.weight1 = nn.Parameter(torch.randn(K, in_planes, 1, kernel_size, kernel_size), requires_grad=True)
        self.weight2 = nn.Parameter(torch.randn(K, in_planes, 1, kernel_size + 2, kernel_size +2), requires_grad=True)
        self.weight3 = nn.Parameter(torch.randn(K, in_planes, 1, kernel_size + 4, kernel_size + 4), requires_grad=True)

        # self.pw1 = nn.Conv2d(self.in_planes, self.out_planes * 11 // 12, kernel_size=1)
        # self.pw2 = nn.Conv2d(self.in_planes, self.out_planes // 24, kernel_size=1)
        # self.pw3 = nn.Conv2d(self.in_planes, self.out_planes // 24, kernel_size=1)

        self.pw1 = nn.Conv2d(self.in_planes, self.out_planes * 11 // 12, kernel_size=1)
        self.pw2 = nn.Conv2d(self.in_planes, self.out_planes // 24, kernel_size=1)
        self.pw3 = nn.Conv2d(self.in_planes, self.out_planes // 24, kernel_size=1)

        self.pw4 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=1)


        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.bn3 = nn.BatchNorm2d(in_planes)

        self.bn4 = nn.BatchNorm2d(out_planes * 11  // 12)
        self.bn5 = nn.BatchNorm2d(out_planes // 24)
        self.bn6 = nn.BatchNorm2d(out_planes // 24)


        if bias:
            self.bias1 = nn.Parameter(torch.Tensor(K, in_planes))
            self.bias2 = nn.Parameter(torch.Tensor(K, in_planes))
            self.bias3 = nn.Parameter(torch.Tensor(K, in_planes))
        else:
            self.bias1 = None
            self.bias2 = None
            self.bias3 = None
        if init_weight:
            self._initialize_weights()
            self.init_weights()



        #TODO 初始化
    def _initialize_weights(self):
        print("init")
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight1[i])
            nn.init.kaiming_uniform_(self.weight2[i])
            nn.init.kaiming_uniform_(self.weight3[i])
        nn.init.normal_(self.pw1.weight, std=0.001)

        nn.init.constant_(self.pw1.bias, 0)
        nn.init.normal_(self.pw2.weight, std=0.001)

        nn.init.constant_(self.pw2.bias, 0)

        nn.init.normal_(self.pw3.weight, std=0.001)

        nn.init.constant_(self.pw3.bias, 0)
        nn.init.normal_(self.pw4.weight, std=0.001)

        nn.init.constant_(self.pw4.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)

        nn.init.constant_(self.bn4.weight, 1)
        nn.init.constant_(self.bn4.bias, 0)
        nn.init.constant_(self.bn5.weight, 1)
        nn.init.constant_(self.bn5.bias, 0)
        nn.init.constant_(self.bn6.weight, 1)
        nn.init.constant_(self.bn6.bias, 0)
        # nn.init.constant_(self.bn7.weight, 1)
        # nn.init.constant_(self.bn7.bias, 0)


    def init_weights(self, pretrained=''):
        print('init module')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        res = x
        batch_size, in_planes, height, width = x.size()
        x1 = x

        softmax_attention1 = self.attention1(x1)

        x1 = x1.reshape(1, -1, height, width)# 变化成一个维度进行组卷积
        weight1 = self.weight1.view(self.K, -1)

        self.groups = in_planes
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight1 = torch.mm(softmax_attention1, weight1).view(-1, 1, self.kernel_size, self.kernel_size)
        if self.bias1 is not None:
            aggregate_bias1 = torch.mm(softmax_attention1, self.bias1).view(-1)
            output1 = F.conv2d(x1, weight=aggregate_weight1, bias=aggregate_bias1, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output1 = F.conv2d(x1, weight=aggregate_weight1, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups= self.groups * batch_size)

        output1 = output1.view(batch_size, self.in_planes, x.size(-2), x.size(-1))

        batch_size, in_planes, height, width = x.size()

        x2 = x

        softmax_attention2 = self.attention2(x2)

        x2 = x2.reshape(1, -1, height, width)  # 变化成一个维度进行组卷积
        weight2 = self.weight2.view(self.K, -1)
        self.groups = in_planes
        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight2 = torch.mm(softmax_attention2, weight2).view(-1, 1, self.kernel_size + 2,
                                                                       self.kernel_size + 2)
        if self.bias2 is not None:
            aggregate_bias2 = torch.mm(softmax_attention2, self.bias2).view(-1)
            output2 = F.conv2d(x2, weight=aggregate_weight2, bias=aggregate_bias2, stride=self.stride,
                               padding=self.padding + 1,
                               dilation=self.dilation, groups=self.in_planes * self.batch_size)
        else:
            output2 = F.conv2d(x2, weight=aggregate_weight2, bias=None, stride=self.stride, padding=2,
                               dilation=1, groups=self.groups * batch_size)

        output2 = output2.view(batch_size, in_planes, x.size(-2), x.size(-1))

        batch_size, in_planes, height, width = x.size()

        x3 = x
        softmax_attention3 = self.attention3(x3)

        x3 = x3.reshape(1, -1, height, width)  # 变化成一个维度进行组卷积

        weight3 = self.weight3.view(self.K, -1)

        self.groups = in_planes

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight3 = torch.mm(softmax_attention3, weight3).view(-1, 1, self.kernel_size + 4,
                                                                       self.kernel_size + 4)
        if self.bias3 is not None:
            aggregate_bias3 = torch.mm(softmax_attention3, self.bias3).view(-1)
            output3 = F.conv2d(x3, weight=aggregate_weight3, bias=aggregate_bias3, stride=self.stride,
                               padding=self.padding + 2,
                               dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output3 = F.conv2d(x3, weight=aggregate_weight3, bias=None, stride=self.stride, padding=3,
                               dilation=self.dilation, groups=self.groups * batch_size)
        output3 = output3.view(batch_size, in_planes, x.size(-2), x.size(-1))

        output1 = self.relu(self.bn1(output1))
        output2 = self.relu(self.bn2(output2))
        output3 = self.relu(self.bn3(output3))

        output1 = self.pw3(output1)
        output1 = self.relu(self.bn6(output1))

        output2 = self.pw2(output2)
        output2 = self.relu(self.bn5(output2))

        output3 = self.pw1(output3)
        output3 = self.relu(self.bn4(output3))

        output = torch.concat([output3,output2,output1],dim=1)
        output = self.pw4(output)
        output = output + res
        return output