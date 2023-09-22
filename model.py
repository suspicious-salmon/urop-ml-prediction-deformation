import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class AutoEncoder(torch.nn.Module):
    def __init__(self, n_input_channels, do_batchnorm=False, act=torch.nn.ReLU()):
        super().__init__()

        self.do_batchnorm = do_batchnorm
        self.do_bias = not do_batchnorm # biases not needed if batchnorm is used
        self.act = act # activation function

        self.n_input_channels = n_input_channels
        self.make_encode()
        self.make_decode()

    def make_encode(self):
        self.conv1 = nn.Conv2d(self.n_input_channels, 512, kernel_size=3, padding="same", bias=self.do_bias)
        self.norm1 = torch.nn.BatchNorm2d(512)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding="same", bias=self.do_bias)
        self.norm2 = torch.nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding="same", bias=self.do_bias)
        self.norm3 = torch.nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def make_decode(self):
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding="same", bias=self.do_bias)
        self.norm4 = torch.nn.BatchNorm2d(256)
        self.upconv4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding="same", bias=self.do_bias)
        self.norm5 = torch.nn.BatchNorm2d(256)
        self.upconv5 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding="same", bias=self.do_bias)
        self.norm6 = torch.nn.BatchNorm2d(512)
        self.upconv6 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.outconv = nn.Conv2d(512, 1, kernel_size=3, padding="same", bias=self.do_bias)

    def encode(self, x):
        x1 = self.conv1(x)
        if self.do_batchnorm: x1 = self.norm1(x1)
        x1 = self.pool1(self.act(x1))

        x2 = self.conv2(x1)
        if self.do_batchnorm: x2 = self.norm2(x2)
        x2 = self.pool2(self.act(x2))

        x3 = self.conv3(x2)
        if self.do_batchnorm: x3 = self.norm3(x3)
        encoded = self.pool3(self.act(x3))

        return encoded
    
    def decode(self, encoded):
        y4 = self.act(self.conv4(encoded))
        if self.do_batchnorm: y4 = self.norm4(y4)
        y4 = self.upconv4(y4)

        y5 = self.act(self.conv5(y4))
        if self.do_batchnorm: y5 = self.norm5(y5)
        y5 = self.upconv5(y5)

        y6 = self.act(self.conv6(y5))
        if self.do_batchnorm: y6 = self.norm6(y6)
        y6 = self.upconv6(y6)

        # return sigmoid(self.outconv(y6)) # not needed with cross-entropy loss function
        return self.outconv(y6)

    def forward(self, x):
        out = self.decode(self.encode(x))
        return out.squeeze(1)
    
class MultiScaleAutoEncoder(torch.nn.Module):
    def __init__(self, n_scales, do_batchnorm=True):
        super().__init__()

        self.n_scales = n_scales
        # in INCREASING image scale!
        self.module_list = torch.nn.ModuleList([AutoEncoder(
            n_input_channels = 1 if i==0 else 2,
            do_batchnorm = do_batchnorm
        ) for i in range(n_scales)])

    def forward(self, input_scaled_images, models):
        output_scaled_images = []
        # loops though input in INCREASING image scale
        for idx, module, img in enumerate(zip(self.module_list, input_scaled_images)):
            output_scaled_images.append(
                module.forward(img) if idx==0 else module.forward(torch.cat(img, output_scaled_images[-1]))
            )

        for image in output_scaled_images:
            plt.imshow(image.to(torch.device("cpu")).detach(), cmap="gray")
            plt.show()
        return output_scaled_images