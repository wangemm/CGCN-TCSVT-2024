from __future__ import print_function, division
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear


class AE_6views(nn.Module):

    def __init__(self, n_stacks, n_input, n_z):
        super(AE_6views, self).__init__()
        dims0 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[0] * 0.8)
            linshidim = int(linshidim)
            dims0.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims0.append(linshidim)

        dims1 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[1] * 0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims1.append(linshidim)

        dims2 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[2] * 0.8)
            linshidim = int(linshidim)
            dims2.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims2.append(linshidim)

        dims3 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[3] * 0.8)
            linshidim = int(linshidim)
            dims3.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims3.append(linshidim)

        dims4 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[4] * 0.8)
            linshidim = int(linshidim)
            dims4.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims4.append(linshidim)

        dims5 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[4] * 0.8)
            linshidim = int(linshidim)
            dims5.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims5.append(linshidim)

        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)
        # encoder2
        self.enc2_1 = Linear(n_input[2], dims2[0])
        self.enc2_2 = Linear(dims2[0], dims2[1])
        self.enc2_3 = Linear(dims2[1], dims2[2])
        self.z2_layer = Linear(dims2[2], n_z)
        # encoder3
        self.enc3_1 = Linear(n_input[3], dims3[0])
        self.enc3_2 = Linear(dims3[0], dims3[1])
        self.enc3_3 = Linear(dims3[1], dims3[2])
        self.z3_layer = Linear(dims3[2], n_z)
        # encoder4
        self.enc4_1 = Linear(n_input[4], dims4[0])
        self.enc4_2 = Linear(dims4[0], dims4[1])
        self.enc4_3 = Linear(dims4[1], dims4[2])
        self.z4_layer = Linear(dims4[2], n_z)
        # encoder5
        self.enc5_1 = Linear(n_input[5], dims5[0])
        self.enc5_2 = Linear(dims5[0], dims5[1])
        self.enc5_3 = Linear(dims5[1], dims5[2])
        self.z5_layer = Linear(dims5[2], n_z)

        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])
        # decoder2
        self.dec2_0 = Linear(n_z, n_z)
        self.dec2_1 = Linear(n_z, dims2[2])
        self.dec2_2 = Linear(dims2[2], dims2[1])
        self.dec2_3 = Linear(dims2[1], dims2[0])
        self.x2_bar_layer = Linear(dims2[0], n_input[2])
        # decoder3
        self.dec3_0 = Linear(n_z, n_z)
        self.dec3_1 = Linear(n_z, dims3[2])
        self.dec3_2 = Linear(dims3[2], dims3[1])
        self.dec3_3 = Linear(dims3[1], dims3[0])
        self.x3_bar_layer = Linear(dims3[0], n_input[3])
        # decoder4
        self.dec4_0 = Linear(n_z, n_z)
        self.dec4_1 = Linear(n_z, dims4[2])
        self.dec4_2 = Linear(dims4[2], dims4[1])
        self.dec4_3 = Linear(dims4[1], dims4[0])
        self.x4_bar_layer = Linear(dims4[0], n_input[4])
        # decoder5
        self.dec5_0 = Linear(n_z, n_z)
        self.dec5_1 = Linear(n_z, dims5[2])
        self.dec5_2 = Linear(dims5[2], dims5[1])
        self.dec5_3 = Linear(dims5[1], dims5[0])
        self.x5_bar_layer = Linear(dims5[0], n_input[5])

    def forward(self, x0, x1, x2, x3, x4, x5):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # encoder2
        enc2_h1 = F.relu(self.enc2_1(x2))
        enc2_h2 = F.relu(self.enc2_2(enc2_h1))
        enc2_h3 = F.relu(self.enc2_3(enc2_h2))
        z2 = self.z2_layer(enc2_h3)
        # encoder3
        enc3_h1 = F.relu(self.enc3_1(x3))
        enc3_h2 = F.relu(self.enc3_2(enc3_h1))
        enc3_h3 = F.relu(self.enc3_3(enc3_h2))
        z3 = self.z3_layer(enc3_h3)
        # encoder4
        enc4_h1 = F.relu(self.enc4_1(x4))
        enc4_h2 = F.relu(self.enc4_2(enc4_h1))
        enc4_h3 = F.relu(self.enc4_3(enc4_h2))
        z4 = self.z4_layer(enc4_h3)
        # encoder5
        enc5_h1 = F.relu(self.enc5_1(x5))
        enc5_h2 = F.relu(self.enc5_2(enc5_h1))
        enc5_h3 = F.relu(self.enc5_3(enc5_h2))
        z5 = self.z5_layer(enc5_h3)
        # add directly
        z = (z0 + z1 + z2 + z3 + z4 + z5) / 6
        # decoder0
        r0 = F.relu(self.dec0_0(z))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)
        # decoder2
        r2 = F.relu(self.dec2_0(z))
        dec2_h1 = F.relu(self.dec2_1(r2))
        dec2_h2 = F.relu(self.dec2_2(dec2_h1))
        dec2_h3 = F.relu(self.dec2_3(dec2_h2))
        x2_bar = self.x2_bar_layer(dec2_h3)
        # decoder3
        r3 = F.relu(self.dec3_0(z))
        dec3_h1 = F.relu(self.dec3_1(r3))
        dec3_h2 = F.relu(self.dec3_2(dec3_h1))
        dec3_h3 = F.relu(self.dec3_3(dec3_h2))
        x3_bar = self.x3_bar_layer(dec3_h3)
        # decoder4
        r4 = F.relu(self.dec4_0(z))
        dec4_h1 = F.relu(self.dec4_1(r4))
        dec4_h2 = F.relu(self.dec4_2(dec4_h1))
        dec4_h3 = F.relu(self.dec4_3(dec4_h2))
        x4_bar = self.x4_bar_layer(dec4_h3)
        # decoder5
        r5 = F.relu(self.dec5_0(z))
        dec5_h1 = F.relu(self.dec5_1(r5))
        dec5_h2 = F.relu(self.dec5_2(dec5_h1))
        dec5_h3 = F.relu(self.dec5_3(dec5_h2))
        x5_bar = self.x5_bar_layer(dec5_h3)

        return x0_bar, x1_bar, x2_bar, x3_bar, x4_bar, x5_bar, z, z0, z1, z2, z3, z4, z5


class AE_5views(nn.Module):

    def __init__(self, n_stacks, n_input, n_z):
        super(AE_5views, self).__init__()
        dims0 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[0] * 0.8)
            linshidim = int(linshidim)
            dims0.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims0.append(linshidim)

        dims1 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[1] * 0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims1.append(linshidim)

        dims2 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[2] * 0.8)
            linshidim = int(linshidim)
            dims2.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims2.append(linshidim)

        dims3 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[3] * 0.8)
            linshidim = int(linshidim)
            dims3.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims3.append(linshidim)

        dims4 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[4] * 0.8)
            linshidim = int(linshidim)
            dims4.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims4.append(linshidim)

        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)
        # encoder2
        self.enc2_1 = Linear(n_input[2], dims2[0])
        self.enc2_2 = Linear(dims2[0], dims2[1])
        self.enc2_3 = Linear(dims2[1], dims2[2])
        self.z2_layer = Linear(dims2[2], n_z)
        # encoder3
        self.enc3_1 = Linear(n_input[3], dims3[0])
        self.enc3_2 = Linear(dims3[0], dims3[1])
        self.enc3_3 = Linear(dims3[1], dims3[2])
        self.z3_layer = Linear(dims3[2], n_z)
        # encoder4
        self.enc4_1 = Linear(n_input[4], dims4[0])
        self.enc4_2 = Linear(dims4[0], dims4[1])
        self.enc4_3 = Linear(dims4[1], dims4[2])
        self.z4_layer = Linear(dims4[2], n_z)

        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])
        # decoder2
        self.dec2_0 = Linear(n_z, n_z)
        self.dec2_1 = Linear(n_z, dims2[2])
        self.dec2_2 = Linear(dims2[2], dims2[1])
        self.dec2_3 = Linear(dims2[1], dims2[0])
        self.x2_bar_layer = Linear(dims2[0], n_input[2])
        # decoder3
        self.dec3_0 = Linear(n_z, n_z)
        self.dec3_1 = Linear(n_z, dims3[2])
        self.dec3_2 = Linear(dims3[2], dims3[1])
        self.dec3_3 = Linear(dims3[1], dims3[0])
        self.x3_bar_layer = Linear(dims3[0], n_input[3])
        # decoder4
        self.dec4_0 = Linear(n_z, n_z)
        self.dec4_1 = Linear(n_z, dims4[2])
        self.dec4_2 = Linear(dims4[2], dims4[1])
        self.dec4_3 = Linear(dims4[1], dims4[0])
        self.x4_bar_layer = Linear(dims4[0], n_input[4])

    def forward(self, x0, x1, x2, x3, x4):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # encoder2
        enc2_h1 = F.relu(self.enc2_1(x2))
        enc2_h2 = F.relu(self.enc2_2(enc2_h1))
        enc2_h3 = F.relu(self.enc2_3(enc2_h2))
        z2 = self.z2_layer(enc2_h3)
        # encoder3
        enc3_h1 = F.relu(self.enc3_1(x3))
        enc3_h2 = F.relu(self.enc3_2(enc3_h1))
        enc3_h3 = F.relu(self.enc3_3(enc3_h2))
        z3 = self.z3_layer(enc3_h3)
        # encoder4
        enc4_h1 = F.relu(self.enc4_1(x4))
        enc4_h2 = F.relu(self.enc4_2(enc4_h1))
        enc4_h3 = F.relu(self.enc4_3(enc4_h2))
        z4 = self.z4_layer(enc4_h3)
        # add directly
        z = (z0 + z1 + z2 + z3 + z4) / 5
        # decoder0
        r0 = F.relu(self.dec0_0(z))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)
        # decoder2
        r2 = F.relu(self.dec2_0(z))
        dec2_h1 = F.relu(self.dec2_1(r2))
        dec2_h2 = F.relu(self.dec2_2(dec2_h1))
        dec2_h3 = F.relu(self.dec2_3(dec2_h2))
        x2_bar = self.x2_bar_layer(dec2_h3)
        # decoder3
        r3 = F.relu(self.dec3_0(z))
        dec3_h1 = F.relu(self.dec3_1(r3))
        dec3_h2 = F.relu(self.dec3_2(dec3_h1))
        dec3_h3 = F.relu(self.dec3_3(dec3_h2))
        x3_bar = self.x3_bar_layer(dec3_h3)
        # decoder4
        r4 = F.relu(self.dec4_0(z))
        dec4_h1 = F.relu(self.dec4_1(r4))
        dec4_h2 = F.relu(self.dec4_2(dec4_h1))
        dec4_h3 = F.relu(self.dec4_3(dec4_h2))
        x4_bar = self.x4_bar_layer(dec4_h3)

        return x0_bar, x1_bar, x2_bar, x3_bar, x4_bar, z, z0, z1, z2, z3, z4


class AE_4views(nn.Module):

    def __init__(self, n_stacks, n_input, n_z):
        super(AE_4views, self).__init__()
        dims0 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[0] * 0.8)
            linshidim = int(linshidim)
            dims0.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims0.append(linshidim)

        dims1 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[1] * 0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims1.append(linshidim)

        dims2 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[2] * 0.8)
            linshidim = int(linshidim)
            dims2.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims2.append(linshidim)

        dims3 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[3] * 0.8)
            linshidim = int(linshidim)
            dims3.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims3.append(linshidim)

        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)
        # encoder2
        self.enc2_1 = Linear(n_input[2], dims2[0])
        self.enc2_2 = Linear(dims2[0], dims2[1])
        self.enc2_3 = Linear(dims2[1], dims2[2])
        self.z2_layer = Linear(dims2[2], n_z)
        # encoder3
        self.enc3_1 = Linear(n_input[3], dims3[0])
        self.enc3_2 = Linear(dims3[0], dims3[1])
        self.enc3_3 = Linear(dims3[1], dims3[2])
        self.z3_layer = Linear(dims3[2], n_z)

        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])
        # decoder2
        self.dec2_0 = Linear(n_z, n_z)
        self.dec2_1 = Linear(n_z, dims2[2])
        self.dec2_2 = Linear(dims2[2], dims2[1])
        self.dec2_3 = Linear(dims2[1], dims2[0])
        self.x2_bar_layer = Linear(dims2[0], n_input[2])
        # decoder3
        self.dec3_0 = Linear(n_z, n_z)
        self.dec3_1 = Linear(n_z, dims3[2])
        self.dec3_2 = Linear(dims3[2], dims3[1])
        self.dec3_3 = Linear(dims3[1], dims3[0])
        self.x3_bar_layer = Linear(dims3[0], n_input[3])

    def forward(self, x0, x1, x2, x3):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # encoder2
        enc2_h1 = F.relu(self.enc2_1(x2))
        enc2_h2 = F.relu(self.enc2_2(enc2_h1))
        enc2_h3 = F.relu(self.enc2_3(enc2_h2))
        z2 = self.z2_layer(enc2_h3)
        # encoder3
        enc3_h1 = F.relu(self.enc3_1(x3))
        enc3_h2 = F.relu(self.enc3_2(enc3_h1))
        enc3_h3 = F.relu(self.enc3_3(enc3_h2))
        z3 = self.z3_layer(enc3_h3)
        # add directly
        z = (z0 + z1 + z2 + z3) / 4
        # decoder0
        r0 = F.relu(self.dec0_0(z))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)
        # decoder2
        r2 = F.relu(self.dec2_0(z))
        dec2_h1 = F.relu(self.dec2_1(r2))
        dec2_h2 = F.relu(self.dec2_2(dec2_h1))
        dec2_h3 = F.relu(self.dec2_3(dec2_h2))
        x2_bar = self.x2_bar_layer(dec2_h3)
        # decoder3
        r3 = F.relu(self.dec3_0(z))
        dec3_h1 = F.relu(self.dec3_1(r3))
        dec3_h2 = F.relu(self.dec3_2(dec3_h1))
        dec3_h3 = F.relu(self.dec3_3(dec3_h2))
        x3_bar = self.x3_bar_layer(dec3_h3)

        return x0_bar, x1_bar, x2_bar, x3_bar, z, z0, z1, z2, z3


class AE_3views(nn.Module):

    def __init__(self, n_stacks, n_input, n_z):
        super(AE_3views, self).__init__()

        dims0 = [n_input[0], n_input[0], 1024]
        dims1 = [n_input[1], n_input[0], 1024]
        dims2 = [n_input[1], n_input[0], 1024]

        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)
        # encoder2
        self.enc2_1 = Linear(n_input[2], dims2[0])
        self.enc2_2 = Linear(dims2[0], dims2[1])
        self.enc2_3 = Linear(dims2[1], dims2[2])
        self.z2_layer = Linear(dims2[2], n_z)

        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])
        # decoder2
        self.dec2_0 = Linear(n_z, n_z)
        self.dec2_1 = Linear(n_z, dims2[2])
        self.dec2_2 = Linear(dims2[2], dims2[1])
        self.dec2_3 = Linear(dims2[1], dims2[0])
        self.x2_bar_layer = Linear(dims2[0], n_input[2])

    def forward(self, x0, x1, x2):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # encoder2
        enc2_h1 = F.relu(self.enc2_1(x2))
        enc2_h2 = F.relu(self.enc2_2(enc2_h1))
        enc2_h3 = F.relu(self.enc2_3(enc2_h2))
        z2 = self.z2_layer(enc2_h3)
        # decoder0
        r0 = F.relu(self.dec0_0(z0))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z1))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)
        # decoder2
        r2 = F.relu(self.dec2_0(z2))
        dec2_h1 = F.relu(self.dec2_1(r2))
        dec2_h2 = F.relu(self.dec2_2(dec2_h1))
        dec2_h3 = F.relu(self.dec2_3(dec2_h2))
        x2_bar = self.x2_bar_layer(dec2_h3)

        return x0_bar, x1_bar, x2_bar, z0, z1, z2


class AE_2views(nn.Module):

    def __init__(self, n_stacks, n_input, n_z):
        super(AE_2views, self).__init__()
        dims0 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[0] * 0.8)
            linshidim = int(linshidim)
            dims0.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims0.append(linshidim)

        dims1 = []
        for idim in range(n_stacks - 2):
            linshidim = round(n_input[1] * 0.8)
            linshidim = int(linshidim)
            dims1.append(linshidim)
        linshidim = 1500
        linshidim = int(linshidim)
        dims1.append(linshidim)

        dims0 = [n_input[0], n_input[0], 1024]
        dims1 = [n_input[1], n_input[0], 1024]

        # encoder0
        self.enc0_1 = Linear(n_input[0], dims0[0])
        self.enc0_2 = Linear(dims0[0], dims0[1])
        self.enc0_3 = Linear(dims0[1], dims0[2])
        self.z0_layer = Linear(dims0[2], n_z)
        # encoder1
        self.enc1_1 = Linear(n_input[1], dims1[0])
        self.enc1_2 = Linear(dims1[0], dims1[1])
        self.enc1_3 = Linear(dims1[1], dims1[2])
        self.z1_layer = Linear(dims1[2], n_z)

        # decoder0
        self.dec0_0 = Linear(n_z, n_z)
        self.dec0_1 = Linear(n_z, dims0[2])
        self.dec0_2 = Linear(dims0[2], dims0[1])
        self.dec0_3 = Linear(dims0[1], dims0[0])
        self.x0_bar_layer = Linear(dims0[0], n_input[0])
        # decoder1
        self.dec1_0 = Linear(n_z, n_z)
        self.dec1_1 = Linear(n_z, dims1[2])
        self.dec1_2 = Linear(dims1[2], dims1[1])
        self.dec1_3 = Linear(dims1[1], dims1[0])
        self.x1_bar_layer = Linear(dims1[0], n_input[1])

    def forward(self, x0, x1):
        # encoder0
        enc0_h1 = F.relu(self.enc0_1(x0))
        enc0_h2 = F.relu(self.enc0_2(enc0_h1))
        enc0_h3 = F.relu(self.enc0_3(enc0_h2))
        z0 = self.z0_layer(enc0_h3)
        # encoder1
        enc1_h1 = F.relu(self.enc1_1(x1))
        enc1_h2 = F.relu(self.enc1_2(enc1_h1))
        enc1_h3 = F.relu(self.enc1_3(enc1_h2))
        z1 = self.z1_layer(enc1_h3)
        # decoder0
        r0 = F.relu(self.dec0_0(z0))
        dec0_h1 = F.relu(self.dec0_1(r0))
        dec0_h2 = F.relu(self.dec0_2(dec0_h1))
        dec0_h3 = F.relu(self.dec0_3(dec0_h2))
        x0_bar = self.x0_bar_layer(dec0_h3)
        # decoder1
        r1 = F.relu(self.dec1_0(z1))
        dec1_h1 = F.relu(self.dec1_1(r1))
        dec1_h2 = F.relu(self.dec1_2(dec1_h1))
        dec1_h3 = F.relu(self.dec1_3(dec1_h2))
        x1_bar = self.x1_bar_layer(dec1_h3)

        return x0_bar, x1_bar, z0, z1


class AE2(nn.Module):
    def __init__(self, n_stacks, n_input, n_z):
        super(AE2, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(n_input[0], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, n_z),
            nn.BatchNorm1d(n_z),
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(n_input[1], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, n_z),
            nn.BatchNorm1d(n_z),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(
            nn.Linear(n_z, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, n_input[0]),
            nn.BatchNorm1d(n_input[0]),
            nn.ReLU(True)
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(n_z, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, n_input[1]),
            nn.BatchNorm1d(n_input[1]),
            nn.ReLU(True)
        )

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        x0b = self.decoder0(h0)
        x1b = self.decoder1(h1)
        return x0b, x1b, h0, h1
