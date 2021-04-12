import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class down(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Average Pooling --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels, filterSize):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used as input and output channels for the
                second convolutional layer.
            filterSize : int
                filter size for the convolution filter. input N would create
                a N x N filter.
        """

        super(down, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(
            inChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )
        self.conv2 = nn.Conv2d(
            outChannels,
            outChannels,
            filterSize,
            stride=1,
            padding=int((filterSize - 1) / 2),
        )

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Average pooling with kernel size 2 (2 x 2).
        x = F.avg_pool2d(x, 2)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        return x


class up(nn.Module):
    """
    A class for creating neural network blocks containing layers:
    
    Bilinear interpolation --> Convlution + Leaky ReLU --> Convolution + Leaky ReLU
    
    This is used in the UNet Class to create a UNet like NN architecture.

    ...

    Methods
    -------
    forward(x, skpCn)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the first convolutional layer.
            outChannels : int
                number of output channels for the first convolutional layer.
                This is also used for setting input and output channels for
                the second convolutional layer.
        """

        super(up, self).__init__()
        # Initialize convolutional layers.
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, stride=1, padding=1)
        # (2 * outChannels) is used for accommodating skip connection.
        self.conv2 = nn.Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)

    def forward(self, x, skpCn):
        """
        Returns output tensor after passing input `x` to the neural network
        block.

        Parameters
        ----------
            x : tensor
                input to the NN block.
            skpCn : tensor
                skip connection input to the NN block.

        Returns
        -------
            tensor
                output of the NN block.
        """

        # Bilinear interpolation with scaling 2.
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        # Convolution + Leaky ReLU
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        # Convolution + Leaky ReLU on (`x`, `skpCn`)
        x = F.leaky_relu(self.conv2(torch.cat((x, skpCn), 1)), negative_slope=0.1)
        return x


class UNet(nn.Module):
    """
    A class for creating UNet like architecture as specified by the
    Super SloMo paper.
    
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `x` to the neural network
        block.
    """

    def __init__(self, inChannels, outChannels):
        """
        Parameters
        ----------
            inChannels : int
                number of input channels for the UNet.
            outChannels : int
                number of output channels for the UNet.
        """

        super(UNet, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = nn.Conv2d(inChannels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, outChannels, 3, stride=1, padding=1)

    def forward(self, x):
        """
        Returns output tensor after passing input `x` to the neural network.

        Parameters
        ----------
            x : tensor
                input to the UNet.

        Returns
        -------
            tensor
                output of the UNet.
        """

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)
        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.

    This is used for backwarping to an image:

    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).

    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """

    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """

        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2 * (x / self.W - 0.5)
        y = 2 * (y / self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        return imgOut


# Creating an array of `t` values for the 7 intermediate frames between
# reference frames I0 and I1.
t = np.linspace(0.125, 0.875, 7)


def getFlowCoeff(indices, device):
    """
    Gets flow coefficients used for calculating intermediate optical
    flows from optical flows between I0 and I1: F_0_1 and F_1_0.

    F_t_0 = C00 x F_0_1 + C01 x F_1_0
    F_t_1 = C10 x F_0_1 + C11 x F_1_0

    where,
    C00 = -(1 - t) x t
    C01 = t x t
    C10 = (1 - t) x (1 - t)
    C11 = -t x (1 - t)

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C00, C01, C10, C11.
    """

    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C11 = C00 = -(1 - (t[ind])) * (t[ind])
    C01 = (t[ind]) * (t[ind])
    C10 = (1 - (t[ind])) * (1 - (t[ind]))
    return (
        torch.Tensor(C00)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C01)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C10)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C11)[None, None, None, :].permute(3, 0, 1, 2).to(device),
    )


def getWarpCoeff(indices, device):
    """
    Gets coefficients used for calculating final intermediate 
    frame `It_gen` from backwarped images using flows F_t_0 and F_t_1.

    It_gen = (C0 x V_t_0 x g_I_0_F_t_0 + C1 x V_t_1 x g_I_1_F_t_1) / (C0 x V_t_0 + C1 x V_t_1)

    where,
    C0 = 1 - t
    C1 = t

    V_t_0, V_t_1 --> visibility maps
    g_I_0_F_t_0, g_I_1_F_t_1 --> backwarped intermediate frames

    Parameters
    ----------
        indices : tensor
            indices corresponding to the intermediate frame positions
            of all samples in the batch.
        device : device
                computation device (cpu/cuda). 

    Returns
    -------
        tensor
            coefficients C0 and C1.
    """

    # Convert indices tensor to numpy array
    ind = indices.detach().numpy()
    C0 = 1 - t[ind]
    C1 = t[ind]
    return (
        torch.Tensor(C0)[None, None, None, :].permute(3, 0, 1, 2).to(device),
        torch.Tensor(C1)[None, None, None, :].permute(3, 0, 1, 2).to(device),
    )


class Slomo(nn.Module):
    def __init__(self, H, W, device):
        super(Slomo, self).__init__()
        self.flowComp = UNet(6, 4)
        self.ArbTimeFlowIntrp = UNet(20, 5)
        self.trainFlowBackWarp = backWarp(H, W, device)
        self.validationFlowBackWarp = backWarp(H, W, device)
        self.device = device

    def forward(self, x, pred_only=False, isTrain=True):
        data, frameidx = x
        I0, IFrame, I1 = data
        I0 = I0.to(self.device)
        I1 = I1.to(self.device)
        IFrame = IFrame.to(self.device)
        flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
        F_0_1 = flowOut[:, :2, :, :]
        F_1_0 = flowOut[:, 2:, :, :]
        fCoeff = getFlowCoeff(frameidx, self.device)
        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0
        if isTrain:
            g_I0_F_t_0 = self.trainFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.trainFlowBackWarp(I1, F_t_1)
        else:
            g_I0_F_t_0 = self.validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = self.validationFlowBackWarp(I1, F_t_1)
        intrpOut = self.ArbTimeFlowIntrp(
            torch.cat(
                (I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1
            )
        )
        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1 = 1 - V_t_0
        if isTrain:
            g_I0_F_t_0_f = self.trainFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = self.trainFlowBackWarp(I1, F_t_1_f)
        else:
            g_I0_F_t_0_f = self.validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = self.validationFlowBackWarp(I1, F_t_1_f)
        wCoeff = getWarpCoeff(frameidx, self.device)
        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
            wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1
        )
        if isTrain:
            if pred_only:
                return Ft_p
            else:
                return (
                    Ft_p,
                    I0,
                    IFrame,
                    I1,
                    g_I0_F_t_0,
                    g_I1_F_t_1,
                    self.trainFlowBackWarp(I0, F_1_0),
                    self.trainFlowBackWarp(I1, F_0_1),
                    F_1_0,
                    F_0_1,
                )
        else:
            if pred_only:
                return Ft_p
            else:
                return (
                    Ft_p,
                    I0,
                    IFrame,
                    I1,
                    g_I0_F_t_0,
                    g_I1_F_t_1,
                    self.validationFlowBackWarp(I0, F_1_0),
                    self.validationFlowBackWarp(I1, F_0_1),
                    F_1_0,
                    F_0_1,
                )


class MLP(nn.Module):
    def __init__(self, H_in, W_in, H1, H2, H_out, W_out):
        super(MLP, self).__init__()
        self.L_in = H_in * W_in
        self.L_out = H_out * W_out
        self.linear1 = torch.nn.Linear(self.L_in, H1)
        self.bn1 = torch.nn.BatchNorm1d(3)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.bn2 = torch.nn.BatchNorm1d(3)
        self.linear3 = torch.nn.Linear(H2, self.L_out)
        self.bn3 = torch.nn.BatchNorm1d(3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = self.bn3(x)
        return x


class Slomofc(nn.Module):
    def __init__(self, H, W, device, pretrain_stage):
        super(Slomofc, self).__init__()
        self.slomo = Slomo(H, W, device)  # output of it is (bs, 3, H, W)
        self.H = H
        self.W = W
        self.L = H * W
        self.pretrain_stage = pretrain_stage
        if self.pretrain_stage:
            self.avgpool = torch.nn.AdaptiveAvgPool2d((int(H / 2), int(W / 2)))
            self.mlp = MLP(int(H / 2), int(W / 2), int(self.L / 2), self.L, H, W)

    def forward(self, x, pred_only=True, isTrain=True):
        Ft_p = self.slomo(x, pred_only=pred_only, isTrain=isTrain)
        if self.pretrain_stage:
            Ft_p = self.avgpool(Ft_p)
            bs, c, h, w = Ft_p.shape
            z = Ft_p.reshape(bs, c, h * w)
            z = self.mlp(z)
            return z
        else:
            return Ft_p
