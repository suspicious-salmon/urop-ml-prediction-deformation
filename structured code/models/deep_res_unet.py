import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()

# ----------------------------
# DeepResUnet Model

class ResBlock(nn.Module):
    def __init__(self, in_dim: int, n_input_channels=1, n_output_channels=1, do_batchnorm=False, do_layernorm=False, do_residual=True, act=nn.ReLU()):
        """A block with 2 convolutions and an optional sum residual. Inputs shape (batch_size, n_input_channels, in_dim, in_dim) and outputs shape (batch_size, n_output_channels, in_dim, in_dim)

        Args:
            - `in_dim`: Shape of square images input into the block. For example, if tensors of shape (16,1,256,256) are input, in_dim should be 256.
            - `n_input_channels`: Number of input channels N, where the shape of the input is (*,N,*,*).
            - `n_output_channels`: Number of output channels O, where the shape of the output is (*,O,*,*). If O != N, the dimension change is applied in the first convolution of the ResBlock.
            - `do_batchnorm`: Whether to use batch normalisation. If True, it is applied before both convolutions.
            - `do_layernorm`: Whether to use layer normalisation. If True, it is applied before both convolutions.
            - `do_residual`: Whether to do residual addition in this block (add residual to the output).
            - `act`: The activation function to use. Applied after normalisation and before convolution.
        """

        super().__init__()

        self.act = act # activation function
        self.do_batchnorm = do_batchnorm
        self.do_layernorm = do_layernorm
        self.do_residual = do_residual
        do_bias = not (do_batchnorm or do_layernorm)

        self.bn1 = nn.BatchNorm2d(n_input_channels)
        self.ln1 = nn.LayerNorm((n_input_channels, in_dim, in_dim))
        self.conv1 = nn.Conv2d(n_input_channels, n_output_channels, kernel_size=3, padding="same", bias=do_bias)

        self.bn2 = nn.BatchNorm2d(n_output_channels)
        self.ln2 = nn.LayerNorm((n_output_channels, in_dim, in_dim))
        self.conv2 = nn.Conv2d(n_output_channels, n_output_channels, kernel_size=3, padding="same", bias=do_bias)

    def forward(self, x, add_x):
        if self.do_batchnorm and self.do_batchnorm: x = self.bn1(x)
        elif self.do_layernorm and self.do_layernorm: x = self.ln1(x)
        e1 = self.conv1(self.act(x))
        if self.do_batchnorm: e1 = self.bn2(e1)
        elif self.do_layernorm: e1 = self.ln2(e1)
        e2 = self.conv2(self.act(e1))
        if self.do_residual: e2 += add_x

        return e2
    
class DownResBlock(nn.Module):
    def __init__(self, in_dim: int, n_input_channels=1, do_batchnorm=False, do_layernorm=False, do_residual=True, act=nn.ReLU(), do_pool=True):
        """Wraps ResBlock class, optionally pre-pending it with a MaxPool2d.
        The shape of the input is (*,N,D,D) and the shape of the output is
        - (*,N,D//2,D//2) if `do_pool` is True
        - (*,N,D,D) otherwise

        Args:
            - `in_dim`: Shape of square images input into the block. For example, if tensors of shape (16,1,256,256) are input, in_dim should be 256.
            - `n_input_channels`: Number of input channels N, where the shape of the input is (*,N,*,*).
            - `do_batchnorm`: Whether to use batch normalisation. If True, it is applied before both convolutions.
            - `do_layernorm`: Whether to use layer normalisation. If True, it is applied before both convolutions.
            - `do_residual`: Whether to do residual addition in this block (add residual to the output).
            - `act`: The activation function to use. Applied after normalisation and before convolution.
            - `do_pool`: Whether to apply max pooling before the ResBlock.
        """

        super().__init__()
        self.resblock = ResBlock(in_dim//2 if do_pool else in_dim, n_input_channels, n_input_channels, do_batchnorm, do_layernorm, do_residual, act)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do_pool = do_pool

    def forward(self, x):
        if self.do_pool: x = self.pool(x)
        e = self.resblock.forward(x, x.clone())
        return e
    
class UpResBlock(nn.Module):
    def __init__(self, in_dim: int, n_input_channels=1, do_batchnorm=False, do_layernorm=False, do_residual=True, act=nn.ReLU(), do_upconv=True):
        """Wraps ResBlock class with an initial upconvolution and concatenation with corresponding-image-scale skip connection.
        The shape of the input is (*,N,D,D) and the shape of the output is
        - (*,N,2D,2D) if `do_pool` is True
        - (*,N,D,D) otherwise

        Args:
            - `in_dim`: Shape of square images input into the block. For example, if tensors of shape (16,1,256,256) are input, in_dim should be 256.
            - `n_input_channels`: Number of input channels N, where the shape of the input is (*,N,*,*).
            - `do_batchnorm`: Whether to use batch normalisation. If True, it is applied before both convolutions.
            - `do_layernorm`: Whether to use layer normalisation. If True, it is applied before both convolutions.
            - `do_residual`: Whether to do residual addition in this block (add residual to the output).
            - `act`: The activation function to use. Applied after normalisation and before convolution.
            - `do_upconv`: Whether to apply upconvolution before the ResBlock and concatenation steps.
        """

        super().__init__()
        do_bias = not (do_batchnorm or do_layernorm)

        # print("Up: ", in_dim*2 if do_upconv else in_dim)
        self.resblock = ResBlock(in_dim*2 if do_upconv else in_dim, n_input_channels*2, n_input_channels, do_batchnorm, do_layernorm, do_residual, act)
        self.upconv = nn.ConvTranspose2d(n_input_channels, n_input_channels, kernel_size=2, stride=2, bias=do_bias)
        self.do_upconv = do_upconv

    def forward(self, x, in_skip_x):
        u = self.upconv(x) if self.do_upconv else x
        c = torch.cat([u, in_skip_x], dim=1)
        e = self.resblock.forward(c, u)
        return e

class DeepResUnet(nn.Module):
    def __init__(self, in_dim: int, n_encode_blocks: int, output_upscale_factor=1, n_input_channels=1, do_output_sigmoid=True, do_batchnorm=False, do_layernorm=False, do_residual=True, act=nn.ReLU()):
        """My DeepResUnet model (see diagram in documentation). Inputs and returns shape (batch_size, n_input_channels, in_dim, in_dim)

        Args:
            - `in_dim`: Shape of square images input into the model. For example, if tensors of shape (16,1,256,256) are input, in_dim should be 256.
            - `n_encode_blocks`: Number of DownResBlocks in encoder, also number of UpResBlocks in decoder.
            - `output_upscale_factor (int, optional)`: _description_. Defaults to 1.
            - `n_input_channels`: Number of input channels N, where the shape of the input is (*,N,*,*).
            - `do_output_sigmoid`: Whether to apply sigmoid function to model's output. Should be used with binary cross-entropy and mean square error loss functions, among others.
            - `do_batchnorm`: Whether to use batch normalisation in the model.
            - `do_layernorm`: Whether to use layer normalisation in the model.
            - `do_residual`: Whether to use residual addition in each ResBlock.
            - `act`: The activation function to use.
        """

        super().__init__()

        width = 32

        self.output_upscale_factor = output_upscale_factor
        self.do_output_sigmoid = do_output_sigmoid
        self.do_batchnorm = do_batchnorm
        self.do_layernorm = do_layernorm
        do_bias = not (do_batchnorm or do_layernorm)

        self.first_conv = nn.Conv2d(n_input_channels, n_input_channels*width, kernel_size=3, padding="same", bias=True)
        self.first_bn = nn.BatchNorm2d(n_input_channels*width)
        self.first_ln = nn.LayerNorm((n_input_channels*width, in_dim, in_dim))
        self.second_conv = nn.Conv2d(n_input_channels*width, n_input_channels*width, kernel_size=3, padding="same", bias=do_bias)
        
        self.encode_blocks = nn.ModuleList([DownResBlock(in_dim//(2**min(i,3)), n_input_channels*width, do_batchnorm,do_layernorm, do_residual, act, do_pool=(i<=2)) for i in range(n_encode_blocks)])
        self.decode_blocks = nn.ModuleList([UpResBlock(in_dim//(2**min(i,3)), n_input_channels*width, do_batchnorm,do_layernorm, do_residual, act, do_upconv=(i<=3)) for i in range(n_encode_blocks, 0, -1)])

        self.last_bn = nn.BatchNorm2d(n_input_channels*width)
        self.last_ln = nn.LayerNorm((n_input_channels*width, in_dim, in_dim))
        self.outconv = nn.Conv2d(n_input_channels*width, 1, kernel_size=1)
        if self.output_upscale_factor > 1: self.output_upscale = nn.ConvTranspose2d(1, 1, kernel_size=output_upscale_factor, stride=output_upscale_factor)

    def forward(self, x):
        e1 = self.first_conv(x)
        if self.do_batchnorm: e1 = self.first_bn(e1)
        elif self.do_layernorm: e1 = self.first_ln(e1)
        out = self.second_conv(e1)
        # print(out.shape)

        out_skip_xs = [] # stores outputs of each DownResBlock to use in skip connections
        for idx, module in enumerate(self.encode_blocks):
            # print("Encode: ", idx)
            out_skip_xs.append(out)
            out = module.forward(out)
        for idx, module in enumerate(self.decode_blocks):
            # print("Decode: ", idx)
            out = module.forward(out, out_skip_xs[len(self.encode_blocks)-1-idx])

        if self.do_batchnorm: out = self.last_bn(out)
        elif self.do_layernorm: out = self.last_ln(out)
        decoded = self.outconv(out)
        if self.output_upscale_factor > 1: decoded = self.output_upscale(decoded)
        if self.do_output_sigmoid: decoded = sigmoid(decoded)

        return decoded