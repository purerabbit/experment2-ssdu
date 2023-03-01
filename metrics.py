import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
 
def normalize_zero_to_one(data, eps=0.):
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)

def compute_ssim(gt, pred):
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    pred=pred.squeeze()
    gt=gt.squeeze()
    # pred=normalize_zero_to_one(pred)
    # gt=normalize_zero_to_one(gt)
    ssim=structural_similarity(gt, pred, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,data_range=pred.max()-pred.min())
    # ssim=structural_similarity(gt, pred,data_range=pred.max()-pred.min())
    return ssim 

def compute_psnr(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """
    if type(recon) is torch.Tensor:
        ref, recon = ref.detach().cpu().numpy(), recon.detach().cpu().numpy()
    ref=ref.squeeze()
    recon=recon.squeeze()
    # ref=normalize_zero_to_one(ref)
    # recon=normalize_zero_to_one(recon)
    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10)) 

    return psnr