# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.model_utils import get_activation

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)

def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)



def get_normalization(out_channels, method, dim='3d'):
    if method == 'batch':
        if dim == '1d':
            return nn.BatchNorm1d(out_channels)
        elif dim == '2d':
            return nn.BatchNorm2d(out_channels)
        elif dim == '3d':
            return nn.BatchNorm3d(out_channels)
    elif method == 'instance':
        if dim == '1d':
            return nn.InstanceNorm1d(out_channels)
        elif dim == '2d':
            return nn.InstanceNorm2d(out_channels)
        elif dim == '3d':
            return nn.InstanceNorm2d(out_channels)
    elif method == 'group':
        return nn.GroupNorm(32, out_channels)
    elif method == 'none':
        return nn.Identity()


class conv2d_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, momentum=0.01, activation=nn.ReLU, norm='batch'):
        super(conv2d_bn_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=1)
        self.norm = get_normalization(out_channels, method=norm, dim=2)
        self.activation = activation()

    def forward(self, x):
        # Assuming x comes in with shape (1, C, X, Y, Z)
        x = x.permute(3, 1, 2, 4, 0).squeeze(4)  # (Y, C, X, Z)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = x.unsqueeze(4).permute(4, 1, 2, 0, 3)  # (1, C, X, Y, Z)
        return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, momentum=0.01, activation=nn.ReLU, norm='batch', dim='3d'):
        super(conv_block, self).__init__()

        self.dim = dim

        if dim == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=1)
        elif dim == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=1)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel, padding=1)

        self.norm = get_normalization(out_channels, method=norm, dim='3d')
        self.activation = activation()

    def forward(self, x):  # (B, C, X, Y, Z)
        if self.dim == '1d':
            # Assuming x comes in with shape (1, C, X, Y, Z)
            x = x.permute(2, 3, 1, 4, 0).squeeze(4)  # (X, Y, C, Z)
            dim0 = x.shape[0]
            x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (X*Y, C, Z)
        elif self.dim == '2d':
            x = x.permute(3, 1, 2, 4, 0).squeeze(4)  # (Y, C, X, Z)

        x = self.conv(x)

        if self.dim == '1d':
            x = x.view(dim0, x.shape[0] // dim0, x.shape[1], x.shape[2]).unsqueeze(4)  # (X, Y, C, Z, 1)
            x = x.permute(4, 2, 0, 1, 3)  # (1, C, X, Y, Z)
        elif self.dim == '2d':
            x = x.unsqueeze(4).permute(4, 1, 2, 0, 3)  # (1, C, X, Y, Z)

        x = self.norm(x)
        x = self.activation(x)

        return x


class deconv3d_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_upsample=(2, 2, 2), kernel=4, stride=2, padding=1, momentum=0.01,
                 activation=ACTIVATION, norm='batch', dim='3d'):
        super(deconv3d_bn_block, self).__init__()
        self.dim = dim
        self.up = nn.Upsample(scale_factor=use_upsample)#, mode='trilinear')

        if dim == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1)
        elif dim == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)

        self.norm = get_normalization(out_channels, method=norm, dim='3d')
        self.activation = ACTIVATION()

    def forward(self, x):
        x = self.up(x)

        if self.dim == '1d':
            # Assuming x comes in with shape (1, C, X, Y, Z)
            x = x.permute(2, 3, 1, 4, 0).squeeze(4)  # (X, Y, C, Z)
            dim0 = x.shape[0]
            x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (X*Y, C, Z)
        elif self.dim == '2d':
            x = x.permute(3, 1, 2, 4, 0).squeeze(4)  # (Y, C, X, Z)

        x = self.conv(x)

        if self.dim == '1d':
            x = x.view(dim0, x.shape[0] // dim0, x.shape[1], x.shape[2]).unsqueeze(4)  # (X, Y, C, Z, 1)
            x = x.permute(4, 2, 0, 1, 3)  # (1, C, X, Y, Z)
        elif self.dim == '2d':
            x = x.unsqueeze(4).permute(4, 1, 2, 0, 3)  # (1, C, X, Y, Z)

        x = self.norm(x)
        x = self.activation(x)

        return x


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, norm_type='batch', encode='3d', decode='3d',
                 activation=ACTIVATION, final='tanh', mc=False, residual=None, x2=False):
        super(Generator, self).__init__()

        self.residual = residual

        encode_block = conv_block
        decode_block = conv_block

        self.max3_pool = nn.MaxPool3d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.down0 = nn.Sequential(
            encode_block(n_channels, nf, activation=act, norm='none', dim=encode),
            encode_block(nf, nf, activation=act, norm=norm_type, dim=encode)
        )
        self.down1 = nn.Sequential(
            encode_block(nf, 2 * nf, activation=act, norm=norm_type, dim=encode),
            encode_block(2 * nf, 2 * nf, activation=act, norm=norm_type, dim=encode)
        )
        self.down2 = nn.Sequential(
            encode_block(2 * nf, 4 * nf, activation=act, norm=norm_type, dim=encode),
            nn.Dropout(p=dropout, inplace=False),
            encode_block(4 * nf, 4 * nf, activation=act, norm=norm_type, dim=encode)
        )
        self.down3 = nn.Sequential(
            encode_block(4 * nf, 8 * nf, activation=act, norm=norm_type, dim=encode),
            nn.Dropout(p=dropout, inplace=False),
            encode_block(8 * nf, 8 * nf, activation=act, norm=norm_type, dim=encode)
        )

        self.up3 = deconv3d_bn_block(8 * nf, 4 * nf, activation=act, norm=norm_type, dim=decode)

        self.conv5 = nn.Sequential(
            decode_block(8 * nf, 4 * nf, activation=act, norm=norm_type, dim=decode),
            nn.Dropout(p=dropout, inplace=False),
            decode_block(4 * nf, 4 * nf, activation=act, norm=norm_type, dim=decode)
        )
        self.up2 = deconv3d_bn_block(4 * nf, 2 * nf, activation=act, norm=norm_type, dim=decode)
        self.conv6 = nn.Sequential(

            decode_block(4 * nf, 2 * nf, activation=act, norm=norm_type, dim=decode),
            nn.Dropout(p=dropout, inplace=False),
            decode_block(2 * nf, 2 * nf, activation=act, norm=norm_type, dim=decode)
        )

        self.up1 = deconv3d_bn_block(2 * nf, nf, activation=act, norm=norm_type, dim=decode)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            decode_block(2 * nf, out_channels, activation=final_layer, norm='none', dim=decode)
        )

        self.conv7_g = nn.Sequential(
            decode_block(2 * nf, out_channels, activation=final_layer, norm='none', dim=decode)
        )

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)
        self.decoder = nn.Sequential(self.up3, self.conv5, self.up2, self.conv6, self.up1)

    def forward(self, x, method=None):
        # x (1, C, X, Y, Z)
        if method != 'decode':
            # reshape
            feat = []
            for i in range(len(self.encoder)):
                if i > 0:
                    x = self.max3_pool(x)
                x = self.encoder[i](x)
                feat.append(x)
            if method == 'encode':
                return feat

        if method == 'decode':
            feat = x

        [x0, x1, x2, x3] = feat

        xu3 = self.up3(x3)
        cat3 = torch.cat([xu3, x2], 1)
        x5 = self.conv5(cat3)  # Dropout
        xu2 = self.up2(x5)
        cat2 = torch.cat([xu2, x1], 1)
        x6 = self.conv6(cat2)  # Dropout
        xu1 = self.up1(x6)
        cat1 = torch.cat([xu1, x0], 1)
        x70 = self.conv7_k(cat1)
        x71 = self.conv7_g(cat1)
        #print(x70.shape)

        return {'out0': x70, 'out1': x71}


if __name__ == '__main__':
    from utils.data_utils import print_num_of_parameters
    g = Generator(n_channels=1, norm_type='batch', final='tanh', encode='3d', decode='3d')
    print_num_of_parameters(g)
    f = g(torch.rand(1, 1, 64, 64, 64), method='encode')
    print('features:')
    for ff in f:
        print(ff.shape)
    #f = g(torch.rand(1, 1, 128, 128, 16), method='encode')
    #print(f[-1].shape)
    #upsample = torch.nn.Upsample(size=(16, 16, 16))
    #fup = upsample(f[-1])
    #print(fup.shape)
    out = g(f, method='decode')
    print('outputs:')
    print(out['out0'].shape)

    print(g(torch.rand(1, 1, 64, 64, 64))['out0'].shape)