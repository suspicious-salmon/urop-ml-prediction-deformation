import torch
import torch.nn as nn

import matplotlib.pyplot as plt # only used for debugging

relu = nn.LeakyReLU()
sigmoid = nn.Sigmoid()

class UNet(nn.Module):
    def __init__(self, in_dim, output_upscale_factor=1, n_input_channels=1, do_output_sigmoid=True, do_batchnorm=False, do_layernorm=False):
        """Ivan's UNet model. Inputs and returns shape (batch_size, n_input_channels, in_dim, in_dim)

        Args:
            in_dim (int): Shape of square images being input into the model. For example, if tensors of shape (1,1,256,256) are input, in_dim should be 256.
            do_output_sigmoid (bool, optional): Defaults to True.
            do_batchnorm (bool, optional): Defaults to False.
            do_layernorm (bool, optional): Defaults to False.
        """
        
        print(f"UNet model, do batchnorm: {do_batchnorm}, do layernorm: {do_layernorm}")
        super().__init__()

        self.output_upscale_factor = output_upscale_factor
        self.do_output_sigmoid = do_output_sigmoid
        self.do_batchnorm = do_batchnorm
        self.do_layernorm = do_layernorm
        do_bias = not (do_batchnorm or do_layernorm)

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        self.e11 = nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1, bias=True) # output: [N, 64, H, W]
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=do_batchnorm) # output: [N, 64, H, W]
        self.bn1 = nn.BatchNorm2d(64)
        self.ln1 = nn.LayerNorm((64, in_dim, in_dim))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [N, 64, H/2, W/2]

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=do_bias) # output: [N, 128, H/2, W/2]
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=do_bias) # output: [N, 128, H/2, W/2]
        self.bn2 = nn.BatchNorm2d(128)
        self.ln2 = nn.LayerNorm((128, in_dim//2, in_dim//2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [N, 128, H/4, W/4]

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=do_bias) # output: [N, 256, H/4, W/4]
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=do_bias) # output: [N, 256, H/4, W/4]
        self.bn3 = nn.BatchNorm2d(256)
        self.ln3 = nn.LayerNorm((256, in_dim//4, in_dim//4))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [N, 256, H/8, W/8]

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=do_bias) # output: [N, 512, H/8, W/8]
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=do_bias) # output: [N, 512, H/8, W/8]
        self.bn4 = nn.BatchNorm2d(512)
        self.ln4 = nn.LayerNorm((512, in_dim//8, in_dim//8))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: [N, 512, H/16, W/16]

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=do_bias) # output: [N, 1024, H/16, W/16]
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=do_bias) # output: [N, 1024, H/16, W/16]
        self.bn5 = nn.BatchNorm2d(1024)
        self.ln5 = nn.LayerNorm((1024, in_dim//16, in_dim//16))

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=do_bias)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=do_bias)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=do_bias)
        self.bn4d = nn.BatchNorm2d(512)
        self.ln4d = nn.LayerNorm((512, in_dim//8, in_dim//8))

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=do_bias)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=do_bias)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=do_bias)
        self.bn3d = nn.BatchNorm2d(256)
        self.ln3d = nn.LayerNorm((256, in_dim//4, in_dim//4))

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=do_bias)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=do_bias)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=do_bias)
        self.bn2d = nn.BatchNorm2d(128)
        self.ln2d = nn.LayerNorm((128, in_dim//2, in_dim//2))

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=do_bias)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=do_bias)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=do_bias)
        self.bn1d = nn.BatchNorm2d(64)
        self.ln1d = nn.LayerNorm((64, in_dim, in_dim))

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        if self.output_upscale_factor > 1: self.output_upscale = nn.ConvTranspose2d(1, 1, kernel_size=output_upscale_factor, stride=output_upscale_factor)

    def forward(self, x):
        # Encoder

        xe11 = self.e11(x)
        if self.do_batchnorm: xe11 = self.bn1(xe11)
        elif self.do_layernorm: xe11 = self.ln1(xe11)
        xe11 = relu(xe11)
        xe12 = self.e12(xe11)
        if self.do_batchnorm: xe12 = self.bn1(xe12)
        elif self.do_layernorm: xe12 = self.ln1(xe12)
        xe12 = relu(xe12)
        xp1 = self.pool1(xe12)

        xe21 = self.e21(xp1)
        if self.do_batchnorm: xe21 = self.bn2(xe21)
        elif self.do_layernorm: xe21 = self.ln2(xe21)
        xe21 = relu(xe21)
        xe22 = self.e22(xe21)
        if self.do_batchnorm: xe22 = self.bn2(xe22)
        elif self.do_layernorm: xe22 = self.ln2(xe22)
        xe22 = relu(xe22)
        xp2 = self.pool2(xe22)

        xe31 = self.e31(xp2)
        if self.do_batchnorm: xe31 = self.bn3(xe31)
        elif self.do_layernorm: xe31 = self.ln3(xe31)
        xe31 = relu(xe31)
        xe32 = self.e32(xe31)
        if self.do_batchnorm: xe32 = self.bn3(xe32)
        elif self.do_layernorm: xe32 = self.ln3(xe32)
        xe32 = relu(xe32)
        xp3 = self.pool3(xe32)

        xe41 = self.e41(xp3)
        if self.do_batchnorm: xe41 = self.bn4(xe41)
        elif self.do_layernorm: xe41 = self.ln4(xe41)
        xe41 = relu(xe41)
        xe42 = self.e42(xe41)
        if self.do_batchnorm: xe42 = self.bn4(xe42)
        elif self.do_layernorm: xe42 = self.ln4(xe42)
        xe42 = relu(xe42)
        xp4 = self.pool4(xe42)

        xe51 = self.e51(xp4)
        if self.do_batchnorm: xe51 = self.bn5(xe51)
        elif self.do_layernorm: xe51 = self.ln5(xe51)
        xe51 = relu(xe51)
        xe52 = self.e52(xe51)
        if self.do_batchnorm: xe52 = self.bn5(xe52)
        elif self.do_layernorm: xe52 = self.ln5(xe52)
        encoded = relu(xe52)

        # Decoder

        xu1 = self.upconv1(encoded)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        if self.do_batchnorm: xd11 = self.bn4d(xd11)
        elif self.do_layernorm: xd11 = self.ln4d(xd11)
        xd12 = relu(self.d12(xd11))
        if self.do_batchnorm: xd12 = self.bn4d(xd12)
        elif self.do_layernorm: xd12 = self.ln4d(xd12)

        xu2 = self.upconv2(xd12)
        xu21 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu21))
        if self.do_batchnorm: xd21 = self.bn3d(xd21)
        elif self.do_layernorm: xd21 = self.ln3d(xd21)
        xd22 = relu(self.d22(xd21))
        if self.do_batchnorm: xd22 = self.bn3d(xd22)
        elif self.do_layernorm: xd22 = self.ln3d(xd22)

        xu3 = self.upconv3(xd22)
        xu31 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu31))
        if self.do_batchnorm: xd31 = self.bn2d(xd31)
        elif self.do_layernorm: xd31 = self.ln2d(xd31)
        xd32 = relu(self.d32(xd31))
        if self.do_batchnorm: xd32 = self.bn2d(xd32)
        elif self.do_layernorm: xd32 = self.ln2d(xd32)

        xu4 = self.upconv4(xd32)
        xu41 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu41))
        if self.do_batchnorm: xd41 = self.bn1d(xd41)
        elif self.do_layernorm: xd41 = self.ln1d(xd41)
        xd42 = relu(self.d42(xd41))
        if self.do_batchnorm: xd42 = self.bn1d(xd42)
        elif self.do_layernorm: xd42 = self.ln1d(xd42)

        decoded = self.outconv(xd42)
        if self.output_upscale_factor > 1: decoded = self.output_upscale(decoded)
        if self.do_output_sigmoid: decoded = sigmoid(decoded)

        return decoded