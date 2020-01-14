import argparse
import numpy as np
import torch
from PIL import Image
from scipy.fftpack import dct, idct

class ToCustomTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self):
        pass

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # handle numpy arrays
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            pic = pic/255

            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class TransCropHorizon(object):
    """
    This transformation crops the horizon and fills the cropped area with black
    pixels or delets them completely.
    Args:
        crop_value (float) [0,1]: Percentage of the image to be cropped from
        the total_step
        set_black (bool): sets the cropped pixels black or delets them completely
    """
    def __init__(self, crop_value, set_black=False):
        assert isinstance(set_black, (bool))
        self.set_black = set_black

        if crop_value >= 0 and crop_value < 1:
            self.crop_value = crop_value
        else:
            print('One or more Arg is out of range!')

    def __call__(self, image):
        crop_value = self.crop_value
        set_black = self.set_black
        image_height = image.size[1]
        crop_pixels_from_top = int(round(image_height*crop_value, 0))

        # convert from PIL to np
        image = np.array(image)

        if set_black==True:
            image[:][0:crop_pixels_from_top-1][:] = np.zeros_like(image[:][0:crop_pixels_from_top-1][:])
        else:
            image = image[:][crop_pixels_from_top:-1][:]

        # reconvert again to PIL
        image = Image.fromarray(image)

        return image


class FilterWeights:
    def __init__(self, n):

        self.m = int(n)
        self.n = n**2
        self.f_tau = np.zeros((self.n, self.m, self.m))
        self.f_tau_inv = np.zeros((self.n, self.m, self.m))

        self.compute_weights()
        self.compute_inverse_weights()

    def get_alpha(self, k):
        if k == 1:
            alpha = np.sqrt(1 / self.n)
        else:
            alpha = np.sqrt(2 / self.n)
        return alpha

    def get_beta(self, i, k):
        beta = np.cos((np.pi * (2 * i - 1) * (k - 1)) / (2 * self.n))
        return beta

    def get_v_prime(self, i):
        v_prime = np.zeros((self.m, 1))
        for j in range(self.m):
            alpha = self.get_alpha(j)
            beta = self.get_beta(i, j)
            v_prime[j] = alpha * beta
        return v_prime

    def get_v_hat_prime(self, i):
        v_hat = np.zeros((self.m, 1))
        for j in range(self.m):
            alpha = self.get_alpha(i)
            beta = self.get_beta(j, i)
            v_hat[j] = alpha * beta
        return v_hat

    def get_v_i(self, i, hat=False):
        # Hat is for the inverse
        p = int(np.ceil(i / self.m))
        q_p = i - self.m * np.floor(i / self.m)
        if q_p:
            q = int(q_p)
        else:
            q = self.m
        if not hat:
            v_p_i = self.get_v_prime(p)
            v_q_i = self.get_v_prime(q)
        else:
            v_p_i = self.get_v_hat_prime(p)
            v_q_i = self.get_v_hat_prime(q)

        v_i = np.matmul(v_p_i, np.transpose(v_q_i))
        return v_i

    def compute_weights(self):
        for i in range(self.n):
            self.f_tau[i, :, :] = self.get_v_i(i, hat=False)

    def compute_inverse_weights(self):
        for i in range(self.n):
            self.f_tau_inv[i, :, :] = self.get_v_i(i, hat=True)


class SpectralTransform(torch.nn.Conv2d):
    pass


class SpectralTransformInverse(torch.nn.Conv2d):
    pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('SpectralTransform') != -1:
        spec_filter = FilterWeights(m.kernel_size[0])  # Kernel is square
        print('Shape of filter')
        print(spec_filter.f_tau.shape)
        print("shape of weights")
        print(m.weight.shape)
        m.weight = torch.nn.Parameter(torch.from_numpy(spec_filter.f_tau).expand(m.weight.size()),
                                      requires_grad=False)
    elif classname.find('SpectralTransformInverse') != -1:
        spec_filter = FilterWeights(m.kernel_size[0])  # Kernel is square
        m.weight = torch.nn.Parameter(torch.from_numpy(spec_filter.f_tau_inv).expand(m.weight.size()),
                                      requires_grad=False)

        # TODO make sure the weights do not change!!


def spectral_masking(T):
    T = T.detach().cpu().numpy()
    threshold = -0.1
    p_keep = 1
    mask_tresh = np.abs(T) > np.max(T) ** threshold # questi li tengo, exp treshold bsc log scale
    mask_dropout = np.random.random_sample(T.shape) < p_keep # questi li tengo
    mask = mask_tresh * mask_dropout
    T[~mask] = 0
    return torch.from_numpy(T)


class Reshaper:
    """Reshape tensors on hypercolumn format

    Expect input to be ordered as BxCxHxW and outputs a BxSCxSCxHW where SC = sqrt(C)"""
    def __init__(self, tensor):
        # TODO: fails with batch size!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.original_shape = tensor.shape
        self.batch_size = self.original_shape[0]
        self.n_channels = self.original_shape[1]

        # Check if we can build the 2D filter
        assert self.n_channels == int(np.sqrt(self.n_channels)) ** 2, 'The number of channels is not a square number'
        self.m = int(np.sqrt(self.n_channels))

    def reshape_hypercolumns(self, tensor):
        # Flatten h and w to get what will be the third dimension w*h
        tensor = torch.flatten(tensor, start_dim=2, end_dim=3)
        # Reshape in size sqrt(n) T sqrt(n)
        tensor = tensor.reshape(self.batch_size, self.m, self.m, -1)

        return tensor

    def reshape_back_hypercolumns(self, tensor):
        # Flatten sqrt(n) x sqrt(n) to get what will be the third dimension (channels)
        tensor = torch.flatten(tensor, start_dim=1, end_dim=2)
        # Reshape w*h to get back h x w
        tensor = tensor.reshape(self.original_shape)

        return tensor


def to_spectral(x):
    for b in x:
        for c in b:
            c = torch.from_numpy(dct(dct(c.T.detach().cpu().numpy(), norm='ortho').T, norm='ortho'))
    return x


def to_spatial(x):
    for b in x:
        for c in b:
            c = torch.from_numpy(idct(idct(c.T.detach().cpu().numpy(), norm='ortho').T, norm='ortho'))
    return x


def get_parser():
    # Parser for training settings
    parser = argparse.ArgumentParser()

    parser.add_argument("--host",
                        type=str,
                        default='rudolf',
                        help="local, rudolf or leonhard ")

    parser.add_argument("--gpu",
                        type=int,
                        default=False,
                        help="CUDA:n where n = [0, 1, 2, 3, 4, 5, 6 ,7]")
    parser.add_argument("--workers",
                        type=int,
                        default=1,
                        help="Number of cpu workers [1:8]")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Model to be trained: [VanillaCNN, SpectralDropoutCNN, SpectralDropoutEasyCNN]")
    parser.add_argument("--epochs",
                        type=int,
                        default=1,
                        help="Training epochs")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate")
    parser.add_argument("--dataset",
                        type=str,
                        default='real',
                        help="Sim: 'sim'; real: 'real'; both='sim_real'")
    parser.add_argument("--validation_split",
                        type=float,
                        default=0.2,
                        help="Validation split size")
    parser.add_argument("--image_res",
                        type=int,
                        default=64,
                        help="Resolution of image (not squared, just used for rescaling")
    return parser
