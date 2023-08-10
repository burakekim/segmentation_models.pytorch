from torch import nn
import torch

class occlusion_encode(nn.Module):
    def __init__(self,
                 occlusion_modality,
                 patch_size,
                 batch_size):
        super().__init__()
        self.up_tabular = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)
        self.double_conv_tabular = nn.Sequential(nn.Conv2d(occlusion_modality, 64, kernel_size=3, padding=1, bias=False), #14 as in number of bands
                                                nn.BatchNorm2d(64),
                                                nn.LeakyReLU(inplace=True),
                                                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                                nn.BatchNorm2d(64),
                                                nn.LeakyReLU(inplace=True))
                                                
        self.batch_size = batch_size
        
    def forward(self, importances):
        unsq_importances = importances.unsqueeze(-1).unsqueeze(-1) #### expand(self.batch_size, -1, -1, -1)
        up_impor = self.up_tabular(unsq_importances)
        encoded_conv_impor = self.double_conv_tabular(up_impor.cuda())
        return encoded_conv_impor #up_import idi burasi
