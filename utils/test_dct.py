from scipy.fftpack import dct, idct
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch


def spectral_masking(T):
    T = torch.from_numpy(T)
    threshold = -0.1
    p_keep = 1
    mask_tresh = np.abs(T) > np.max(T.data.numpy()) ** threshold # questi li tengo, exp treshold bsc log scale
    mask_dropout = np.random.random_sample(T.shape) < p_keep # questi li tengo
    mask = mask_tresh * torch.BoolTensor(mask_dropout)
    T[~mask] = 0
    return T.data.numpy()


img_p = ''.join(['/home/gianmarco/spectral-dropout/varia/', 'default_168_sim.jpg'])

img = cv2.imread(img_p,0)/255


fig, axs = plt.subplots(3, 2)
axs[0,0].imshow(img, cmap='gray')


img_dct = dct(dct(img.T, norm='ortho').T, norm='ortho') # img_dct = dct(img, type=2, n=4)

axs[0,1].imshow(np.log(img_dct), cmap='gray')
img_dct_dr = spectral_masking(img_dct.copy())
img_dct_inv = idct(idct(img_dct_dr.T, norm='ortho').T, norm='ortho')
axs[1,0].imshow(np.log(img_dct_dr), cmap='gray')
axs[1,1].imshow(img_dct_inv, cmap='gray')

img_fft = fft2(img)
img_fft_inv = ifft2(img_fft)
axs[2,0].imshow(np.log(np.abs(img_fft)), cmap='gray')
axs[2,1].imshow(np.abs(img_fft_inv), cmap='gray')

for i in range(3):
    for j in range(2):
        axs[i,j].axis('off')
plt.show()

# n, bins, patches = plt.hist(img_dct.reshape(-1,1),bins=10, range=(0,1), log=True)
# plt.show()
#
# n, bins, patches = plt.hist(img_dct_dr.reshape(-1,1),bins=10, range=(0,1), log=True)
# plt.show()