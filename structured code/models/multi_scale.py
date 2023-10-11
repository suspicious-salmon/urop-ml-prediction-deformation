import torch
import torch.nn as nn

class MultiScale(nn.Module):
    def __init__(self, model: nn.Module, scales: list, do_output_sigmoid=True, do_batchnorm=False, do_layernorm=False):
        """Multi-Scale Neural Network class. Takes in list of pytorch tensors, with side lengths of determined by `scales`.
        e.g. If `scales` = (32,128) then input shape is [tensor(B,N,32,32), tensor(B,N,128,128)] (must be in ascending order of scale).

        Args:
            model: model to be used for each scale of the multi scale nn
            scales: list of square image scales being used, for example: (32, 128).
            - `do_output_sigmoid`: Whether to apply sigmoid function to model's output. Should be used with binary cross-entropy and mean square error loss functions, among others.
            - `do_batchnorm`: Whether to use batch normalisation in the model.
            - `do_layernorm`: Whether to use layer normalisation in the model.
        """

        super().__init__()

        scales = sorted(scales) # ensure scales are sorted in increasing image scale
        module_list = []
        # iterate through scales, creating a new instance of the model for each one
        for idx, scale in enumerate(scales):
            module_list.append(model(
                scale,
                output_upscale_factor = int(scales[idx+1]/scale) if idx != len(scales)-1 else 1, # output image needs to be upscaled to feed into next scale size of pyramid
                n_input_channels = 1 if idx==0 else 2, # all but the smallest scale module have 2-channel input (one from the current scale image, one upscaled output from the previous scale)
                do_output_sigmoid = do_output_sigmoid,
                do_batchnorm = do_batchnorm,
                do_layernorm = do_layernorm))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, input_scaled_images):
        output_scaled_images = []
        # loops though input, which must be in increasing image scale
        for idx, (module, img) in enumerate(zip(self.module_list, input_scaled_images)):
            print(img.dtype, img.shape)
            if idx!=0: print(output_scaled_images[-1].dtype, output_scaled_images[-1].shape)
            output_scaled_images.append(module.forward(img) if idx==0 else module.forward(torch.cat([img, output_scaled_images[-1]], dim=1)))

        # for image in output_scaled_images:
        #     plt.imshow(image[0][0].to(torch.device("cpu")).detach(), cmap="gray")
        #     plt.show()
        return output_scaled_images[-1]
