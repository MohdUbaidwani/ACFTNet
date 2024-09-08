
from typing_extensions import Self
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cv2
import torch.fft as fft
import time
import matplotlib.pyplot as plt


#Adaptive fusion block
class GAFP(nn.Module):
    def __init__(self, channels1,channels2, channels_out, m=-0.80):
        super(GAFP, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

        self.project_conv1 = nn.Conv2d(channels1, channels_out//2, kernel_size=1, bias=False)
        self.project_conv2 = nn.Conv2d(channels_out//2, channels1, kernel_size=1, bias=False)

    def forward(self, fea1, fea2):
       
        mix_factor = self.mix_block(self.w)
        conv1 = self.project_conv1(fea1)
        conv2 = self.project_conv2(fea2)
        conv1_mixed = conv1*mix_factor.expand_as(conv1)

        conv2_mixed = conv2*(1-mix_factor.expand_as(conv2))
        gel1 = F.gelu(conv1_mixed)
        gel2 = F.gelu(conv2_mixed)
        gate_mult1 = gel1*conv1_mixed
        gate_mult2 = gel2*conv2_mixed

        out = torch.cat([gate_mult1, gate_mult2], dim=1)

        return out


                                                           
class SkipAtt(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(SkipAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x):

      
        y = self.avg_pool(x)
       
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
      
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN2(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x



class GDFN2(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN2, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)

        self.conv1_1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1,
                              groups=hidden_channels, bias=False)

        self.conv3_3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size= 3, padding=1,
                              groups=hidden_channels, bias=False)


        self.conv5_5 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size= 5, padding=2,
                              groups=hidden_channels, bias=False)

        self.project_out = nn.Conv2d(hidden_channels*2, channels, kernel_size=1, bias=False)
    def forward(self, x):
        x = self.project_in(x)
        x1 = self.conv1_1(x)
        x3 = self.conv3_3(x)
        x5 = self.conv5_5(x)

        gate1 = F.gelu(x1)*x3
        gate2 = F.gelu(x1)*x5

        x = self.project_out(torch.cat([gate1,gate2],axis = 1))

        return x

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class UpSample1(nn.Module):
    def __init__(self, channels):
        super(UpSample1, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class s_feature(nn.Module):
    def __init__(self, channels):
        super(s_feature, self).__init__()
        self.convf1 =nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.convf2 =nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)


    def forward(self, x):
        # frequency
        x=self.convf1(x)
        x2=F.gelu(x)
        x3=self.convf2(x2)
        return x3
#frequncy processing and fusion 
class c_feature(nn.Module):                         
    def __init__(self, channels):
        super(c_feature, self).__init__()
        self.convf1_magnitude = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.convf2_magnitude = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.convf1_phase = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.convf2_phase = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.project_out = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)
        self.gfp=GAFP(channels,channels,channels*2)
    def forward(self, x):
        # Frequency domain representation
        x_fft = fft.fftn(x, dim=(-2, -1))

        # Separate magnitude and phase
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # Process magnitude
        magnitude1 = self.convf1_magnitude(magnitude)
        magnitude2 = F.gelu(magnitude1)
        magnitude3 = self.convf2_magnitude(magnitude2)

        # Process phase
        phase1 = self.convf1_phase(phase)
        phase2 = F.gelu(phase1)
        phase3 = self.convf2_phase(phase2)
        f2=torch.exp(1j * phase3)

        x_fft_processed = magnitude3 * f2
        # Transform back to spatial domain
        qf = fft.ifftn(x_fft_processed, dim=(-2, -1)).real
        fused_features=self.gfp(x,qf)
        f_d=self.project_out(fused_features)
        
        return f_d

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, prelu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.prelu = nn.PReLU() if prelu else None

    def forward(self, x):
        y=x
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.prelu is not None:
            x = self.prelu(x)
        return x+y

class attention_to_1x1(nn.Module):
    def __init__(self, channels):
        super(attention_to_1x1, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels*2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y=x
        x=torch.mean(x,-1)
        x=torch.mean(x ,-1)
        x=torch.unsqueeze(x ,-1)
        x=torch.unsqueeze(x ,-1)
        xx = self.conv2(self.conv1(x))
        xx = self.sigmoid(xx)
        return xx*y

class mymodel(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128], num_refinement=4,
                 expansion_factor=2.66, ch=[64,32,16,64]):
        super(mymodel, self).__init__()

        self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.embed_conv = nn.Conv2d(1, channels[0], kernel_size=3, padding=1, bias=False)
     
        self.encoder1 = nn.Sequential(*[TransformerBlock(channels[0], num_heads[0], expansion_factor) for _ in range(num_blocks[0])])
        self.encoder2 = nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor) for _ in range(num_blocks[1])])
        self.encoder3 = nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor) for _ in range(num_blocks[2])])
        self.encoder4 = nn.Sequential(*[TransformerBlock(channels[3], num_heads[3], expansion_factor) for _ in range(num_blocks[3])])
        
        self.down1 = DownSample(channels[0])
        self.down2 = DownSample(channels[1])
        self.down3 = DownSample(channels[2])

        self.ups_1=UpSample1(128)
        self.ups_2=UpSample1(64)
        self.ups_3=UpSample1(32)
       

        self.Basic=BasicConv(1,16)
        self.skipCArgb2=SkipAtt(64)
        self.skipCAr2=SkipAtt(64)
        self.skipCAg2=SkipAtt(64)
        self.skipCAb2=SkipAtt(64)

        self.skipCArgb1=SkipAtt(32)
        self.skipCAr1=SkipAtt(32)
        self.skipCAg1=SkipAtt(32)
        self.skipCAb1=SkipAtt(32)

        self.skipCArgb0=SkipAtt(16)
        self.skipCAr0=SkipAtt(16)
        self.skipCAg0=SkipAtt(16)
        self.skipCAb0=SkipAtt(16)

        self.feature_r=c_feature(channels[0])
        self.feature_g=c_feature(channels[0])
        self.feature_b=c_feature(channels[0])
        self.feature_r1=s_feature(channels[1])                
                                                               
        self.feature_g1=s_feature(channels[1])                  
        self.feature_b1=s_feature(channels[1])
        self.feature_r2=s_feature(channels[2])
        self.feature_g2=s_feature(channels[2])
        self.feature_b2=s_feature(channels[2])

        self.attentionR1=attention_to_1x1(channels[0])
        self.attentionG1=attention_to_1x1(channels[0])
        self.attentionB1=attention_to_1x1(channels[0])

        self.attentionR2=attention_to_1x1(channels[1])
        self.attentionG2=attention_to_1x1(channels[1])
        self.attentionB2=attention_to_1x1(channels[1])

        self.attentionR3=attention_to_1x1(channels[2])
        self.attentionG3=attention_to_1x1(channels[2])
        self.attentionB3=attention_to_1x1(channels[2])

        self.redux1=nn.Conv2d(4*channels[0], channels[0], kernel_size=1, bias=False)
        self.redux2=nn.Conv2d(4*channels[1] ,channels[1],kernel_size=1, bias=False)
        self.redux3=nn.Conv2d(4*channels[2], channels[2], kernel_size=1, bias=False)

        self.ups1 = UpSample1(32)
        self.reduces2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.reduces1=nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))

        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False)
        self.output1= nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)

        self.ups2 = UpSample1(16)
        self.outputl=nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False)

    def forward(self,RGB_input):
        ###-------encoder for RGB-------####
        fo_r = self.embed_conv(RGB_input[:,0:1,:,:])
       
        fo_g = self.embed_conv(RGB_input[:,1:2,:,:])
        fo_b = self.embed_conv(RGB_input[:,2:3,:,:])

        fo_rgb=self.embed_conv_rgb(RGB_input)

        fo_ro = self.feature_r(fo_r)
       

     
        fo_go = self.feature_g(fo_g)
     
        fo_bo = self.feature_b(fo_b)


        fo_r1 = self.feature_r1(self.down1(fo_ro))
    

        fo_g1 = self.feature_g1(self.down1(fo_go))
      

        fo_b1= self.feature_b1(self.down1(fo_bo))
        

        fo_r2 = self.feature_r2(self.down2(fo_r1))
       

        fo_g2 = self.feature_g2(self.down2(fo_g1))
       

        fo_b2 = self.feature_b2(self.down2(fo_b1))
       

        ###-------Encoder------###
        out_enc_rgb1 = self.redux1(torch.cat([self.encoder1(fo_rgb),self.attentionR1(fo_ro)*self.encoder1(fo_rgb),self.attentionG1(fo_go)*self.encoder1(fo_rgb),self.attentionB1(fo_bo)*self.encoder1(fo_rgb)],dim=1),)
        

     
        out_enc_rgb2 = self.redux2(torch.cat([self.encoder2(self.down1(out_enc_rgb1)),self.attentionR2(fo_r1)*self.encoder2(self.down1(out_enc_rgb1)),self.attentionG2(fo_g1)*self.encoder2(self.down1(out_enc_rgb1)),self.attentionB2(fo_b1)*self.encoder2(self.down1(out_enc_rgb1))],dim=1))
        

        out_enc_rgb3 = self.redux3(torch.cat([self.encoder3(self.down2(out_enc_rgb2)),self.attentionR3(fo_r2)*self.encoder3(self.down2(out_enc_rgb2)),self.attentionG3(fo_g2)*self.encoder3(self.down2(out_enc_rgb2)),self.attentionB3(fo_b2)*self.encoder3(self.down2(out_enc_rgb2))],dim=1))
    
        out_enc_rgb4 = self.encoder4(self.down3(out_enc_rgb3))

        ###-------SKIP CONNECTIONS------###
        skip3=self.redux3(torch.cat([out_enc_rgb3,fo_r2*out_enc_rgb3,fo_g2*out_enc_rgb3,fo_b2*out_enc_rgb3],dim=1))
        skip2=self.redux2(torch.cat([out_enc_rgb2,fo_r1*out_enc_rgb2,fo_g1*out_enc_rgb2,fo_b1*out_enc_rgb2],dim=1))
        skip1=self.redux1(torch.cat([out_enc_rgb1,fo_ro*out_enc_rgb1,fo_go*out_enc_rgb1,fo_bo*out_enc_rgb1],dim=1))
      

     
       
       


        ###-------Dencoder------###
        out_dec3 = self.decoders[0](self.reduces1(torch.cat([(self.ups_1(out_enc_rgb4)), skip3], dim=1)))
       
        out_dec2 = self.decoders[1](self.reduces2(torch.cat([self.ups_2(out_dec3),skip2], dim=1)))
       
        fd = self.decoders[2](torch.cat([self.ups_3(out_dec2),skip1], dim=1))
      
        fr = self.refinement(fd)
        
        out1=self.output(self.outputl(fr))
        return out1


