

#查看spasity是否是可以强固定学习ratio
import torch
import torch.nn as nn
from mri_tools import  ifft2,fft2
from .memc_loupe import Memc_LOUPE

class TOTAL_LOUPE(nn.Module):
    def __init__(self, input_shape, slope, sample_slope, device, sparsity_under=0.25,sparsity_select=0.125):
        super(TOTAL_LOUPE, self).__init__()  #这句话是否有用?
        self.under_Memc_LOUPE_Model = Memc_LOUPE(input_shape, slope=slope, sample_slope=sample_slope, device=device, sparsity=sparsity_under)
        self.select_Memc_LOUPE_Model = Memc_LOUPE(input_shape, slope=slope, sample_slope=sample_slope, device=device, sparsity=sparsity_select)
        
    def forward(self,kspace):

        onemask=torch.ones_like(kspace)
        mask_under=self.under_Memc_LOUPE_Model(onemask)
        mask_select=self.select_Memc_LOUPE_Model(onemask)
        
        mask=mask_under   #0-1
        mask_dc=mask*mask_select  #0-1
        mask_loss=mask-mask_dc  #0-1

        return mask,mask_dc,mask_loss