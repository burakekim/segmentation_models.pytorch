
import json
import os
import random
import ssl
from argparse import ArgumentParser
from datetime import datetime

import albumentations as A
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from .naturalness_dataset import MapInWild_Naturalness
from ..decoders.unetinfluence import UnetInfluence 
from .helpers import index_modalities
from .osm import OcclusionSensitivity
from pytorch_lightning.loggers import TensorBoardLogger

ssl._create_default_https_context = ssl._create_unverified_context

def seed_everything(seed):
    random.seed(seed)
    pl.seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True

class MIW(pl.LightningModule):

    modalities = {'SAR' : ('VV','VH'),
                'Sentinel_2_RGB': ('B2','B3','B4'),
                'Sentinel_2_RGBNIR': ('B2','B3','B4','B5'),
                'Sentinel_2_ALL': ('B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'),
                'ESA_WC': ('ESA_wc'),
                'VIIRS':('avg_rad')}

    all_bands = ["VV","VH","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","ESA_wc","avg_rad"] 

    sar_indexes = index_modalities(all_bands, modalities['SAR'])
    s2_rgb_indexes = index_modalities(all_bands, modalities['Sentinel_2_RGB'])
    s2_rgbnir_indexes = index_modalities(all_bands, modalities['Sentinel_2_RGBNIR'])
    s2_all_indexes = index_modalities(all_bands, modalities['Sentinel_2_ALL'])
    esa_wc_indexes = index_modalities(all_bands, modalities['ESA_WC'])
    viirs_indexes = index_modalities(all_bands, modalities['VIIRS'])

    def __init__(
        self,
        hparams
    ):
        super().__init__()
        #self.hparams.update(hparams)
        self.learning_rate = hparams.lr
        self.num_workers = os.cpu_count()
        self.lr = hparams.lr
        self.dataset_root = hparams.dataset_root
        self.split_file = hparams.split_file 
        self.subset_file = hparams.subset_file 
        self.bands = hparams.bands 
        self.crop_size = hparams.crop_size
        self.batch_size = hparams.batch_size
        self.classes = hparams.classes
        self.clsf_aux_params = hparams.clsf_aux_params
        self.seg_activation = hparams.seg_activation
        self.occlusion_modality = hparams.occlusion_modality
        
        # Instantiate datasets, model, and trainer params if provided      
        train_transform = A.Compose([A.RandomCrop(self.crop_size[0], self.crop_size[1], p=1.0), A.RandomRotate90(p=1)])

        val_transform = A.Compose([A.RandomCrop(self.crop_size[0], self.crop_size[1], p=1.0)])
                
        test_transform = A.Compose([A.RandomCrop(self.crop_size[0], self.crop_size[1], p=1.0)])

        self.train_dataset = MapInWild_Naturalness(split_file= self.split_file, root= self.dataset_root,split='train', 
                                bands= self.bands, subsetpath =  self.subset_file, transforms=train_transform)

        self.val_dataset = MapInWild_Naturalness(split_file= self.split_file, root= self.dataset_root,split='validation', 
                                bands= self.bands, subsetpath =  self.subset_file, transforms=val_transform)
        
        self.test_dataset = MapInWild_Naturalness(split_file= self.split_file, root= self.dataset_root,split='test', 
                                bands= self.bands, subsetpath =  self.subset_file, transforms=test_transform)

        assert set(self.test_dataset.ids).isdisjoint(set(self.train_dataset.ids))
        assert set(self.test_dataset.ids).isdisjoint(set(self.val_dataset.ids))
        assert set(self.val_dataset.ids).isdisjoint(set(self.train_dataset.ids))
        
        self.model = self._prepare_model()

        self.loss_fn = nn.MSELoss()

        self.train_err = torchmetrics.MeanAbsoluteError()
        self.eval_err =  torchmetrics.MeanAbsoluteError()

        self.dict = {} 
        self.list_wdpa = []
        self.list_importance = []

    def forward(self, image:torch.Tensor):
        return self.model(image)

    def training_step(self, batch, batch_nb):
        #https://wandb.ai/borisd13/lightning-kitti/reports/Lightning-Kitti--Vmlldzo3MTcyMw
        #https://www.kaggle.com/code/dhananjay3/image-segmentation-from-scratch-in-pytorch 

        self.model.train()
        torch.set_grad_enabled(True)

        img = batch[0]
        mask = batch[1]
        wdpa_id = batch[2]

        img = img.float()
        mask = mask.float()

        #print("img.shape", img.shape) #(4,14,512,512)
        #occlusion_values = self.band_osm_values(img) ## (14,1)
        occlusion_values = self.modality_osm_values(img) ## (14,1)
        #print("occlusion_val.shape", occlusion_values.shape)
        tensor_occlusion_values = self.prepare_importance_tensor(occlusion_values)

        seg_mask, clsf_score = self.model(img, tensor_occlusion_values) 

        loss_seg = self.loss_fn(seg_mask, mask)
        loss = loss_seg 
        
        tr_reg_im = self.train_err(seg_mask, mask)

        metrics = {"train_loss": loss, "train_err": tr_reg_im}
        
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True)
        return loss
    ###########################################

    def validation_step(self, batch, batch_idx):
        acc, iou, loss = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_iou": iou, "val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        acc, iou, loss = self._test_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _test_step(self, batch, batch_idx):
        #https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
        #https://pytorch-lightning.readthedocs.io/en/stable/extensions/loops.html
        self.model.eval()
        torch.set_grad_enabled(False) 

        img = batch[0]
        mask = batch[1]
        wdpa_id = batch[2]

        img = img.float()
        mask = mask.float()

        #occlusion_values = self.band_osm_values(img) 
        occlusion_values = self.modality_osm_values(img) ## (14,1)

        tensor_occlusion_values = self.prepare_importance_tensor(occlusion_values)
        #####################################
        self.list_wdpa.append(wdpa_id.clone().cpu().detach().numpy().tolist())
        self.list_importance.append(tensor_occlusion_values.clone().cpu().detach().numpy().tolist())
        self.dict['wdpa'] = self.list_wdpa 
        self.dict['occlusion_values'] = self.list_importance
 
        json_object = json.dumps(self.dict)

        with open("./modality_importance_values.json", "w") as outfile:
            outfile.write(json_object)
        ###########################################

        seg_mask, clsf_score = self.model(img, tensor_occlusion_values) 

        loss_seg = self.loss_fn(seg_mask, mask)
        loss = loss_seg 
        
        self.eval_err(seg_mask, mask)

        return self.eval_err, loss

    def _shared_eval_step(self, batch, batch_idx):
        #https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
        #https://pytorch-lightning.readthedocs.io/en/stable/extensions/loops.html
        self.model.eval()
        torch.set_grad_enabled(False) 

        img = batch[0]
        mask = batch[1]
        wdpa_id = batch[2]

        img = img.float()
        mask = mask.float()

        occlusion_values = self.modality_osm_values(img) ## (14,1)
        #print("occlusion_values", occlusion_values)

        tensor_occlusion_values = self.prepare_importance_tensor(occlusion_values)

        seg_mask, clsf_score = self.model(img, tensor_occlusion_values) 

        loss_seg = self.loss_fn(seg_mask, mask)
        loss = loss_seg 
        
        self.eval_err(seg_mask, mask)

        return self.eval_err, loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.0001)
        return opt

    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True)

    def val_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True)

    def test_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True)

    def _prepare_model(self):
        model = UnetInfluence(in_channels=len(self.bands),
                                patch_size=self.crop_size[0],
                                batch_size=self.batch_size,
                                classes=len(self.classes),
                                occlusion_modality=self.occlusion_modality,
                                aux_params=self.clsf_aux_params)
        return model

    def band_osm_values(self, input_osm):
        probs_list = list()
        label = 1
        fill_value = 0

        block_size = (self.crop_size[0],self.crop_size[0])
        input_size = (self.crop_size[0],self.crop_size[0]) 

        batch_size, channels, _, _ = input_osm.shape

        for i in range(len(self.bands)):
            bands = [i]
            interpretor = OcclusionSensitivity(self.model, None, None, input_size, block_size, fill_value, target=label, bands=bands, occlusion_mode='spectral', batch_size=self.batch_size, occlusion_modality=self.occlusion_modality)
            probabilities = interpretor.interpret(input_osm) 
            probs_list.append(probabilities)
        return probs_list 

    def modality_osm_values(self, input_osm):
        probs_list = list()
        label = 1
        fill_value = 0

        block_size = (self.crop_size[0],self.crop_size[0])
        input_size = (self.crop_size[0],self.crop_size[0]) 

        batch_size, channels, _, _ = input_osm.shape

        for modality_index in [self.sar_indexes, self.s2_rgb_indexes, self.s2_rgbnir_indexes, self.s2_all_indexes, self.esa_wc_indexes, self.viirs_indexes]:
            bands = modality_index
            interpretor = OcclusionSensitivity(self.model, None, None, input_size, block_size, fill_value, target=label, bands=bands, occlusion_mode='spectral', batch_size=self.batch_size, occlusion_modality=self.occlusion_modality)
            probabilities = interpretor.interpret(input_osm) 
            print("probabilities",probabilities)
            probs_list.append(probabilities)
        return probs_list 

    # def prepare_importance_tensor(self, probs_list):
    #     """        
    #     For a given list with 14 tensors that consists of 4 items each this function outputs torch.Size([4,14])            
    #     """
    #     stacked_probs = [torch.stack(probs_list[ii]) for ii in range(len(probs_list))]
    #     squez_stacked_probs = torch.stack(stacked_probs).squeeze()
    #     split_probs = torch.split(squez_stacked_probs,list(np.arange(1,self.batch_size+1,1))) #
    #     transposed_probs = [split_probs[i].T for i in range(len(split_probs))]
    #     concat_probs = torch.cat(transposed_probs)
    #     return concat_probs

    def prepare_importance_tensor(self, probs_list):
        """        
        For a given list with 14 tensors that consists of 4 items each this function outputs torch.Size([4,14])            
        """
        probs_list_ = [torch.as_tensor(probs_list[ii]) for ii in range(len(probs_list))]
        squez_stacked_probs = torch.stack(probs_list_).squeeze()
        split_probs = squez_stacked_probs.T
        return split_probs

    # def tuple_to_tensors(self, tuples):
    #     return torch.stack(list(tuples),dim=0)
    
    # def prepare_importance_tensor(self, probs_list):
    #     tensor_list = torch.stack(probs_list)
    #     swaped_occlusion_values = torch.swapaxes(tensor_list, 0, 1)
    #     split_importances = torch.split(swaped_occlusion_values[0],6)
    #     split_importances_tensors = self.tuple_to_tensors(split_importances)
    #     return split_importances_tensors

def main(hparams):
    seed_everything(31)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    miw_model = MIW(hparams=hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_iou", mode="max", verbose=True
    )

    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), name="lightning_logs")

    trainer = pl.Trainer(devices=1, 
                        precision=16,
                        callbacks=[checkpoint_callback],
                        logger=tb_logger,
                        max_epochs=20,
                        log_every_n_steps=3,
                        accelerator="gpu")

    trainer.fit(
        miw_model)

    valid_metrics = trainer.validate(miw_model, dataloaders=miw_model.val_dataloader(), verbose=False)
    print(valid_metrics)
    test_metrics = trainer.test(miw_model, dataloaders=miw_model.test_dataloader(), verbose=False) 
    print(test_metrics)

    logs = valid_metrics + test_metrics
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%H_%M_%S_%f_%b_%d_%Y")
    log_file = open("{}.json".format(timestampStr), "w")
    json.dump(logs, log_file)
    log_file.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", default= '/data/Dataset_')
    parser.add_argument("--split_file", default= '/data/aux_/split_IDs/tvt_split.csv') # if None -> s2_summer
    parser.add_argument("--subset_file", default= '/data/aux_/single_temporal_subset/single_temporal_subset.csv')
    parser.add_argument("--bands", default=("VV","VH","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","2020_Map","avg_rad"))
    parser.add_argument("--crop_size", default=(256,256)) 
    parser.add_argument("--batch_size", default=8) 
    parser.add_argument("--classes", default=['naturalness'])
    parser.add_argument("--lr", default=1e-2)
    parser.add_argument("--occlusion_modality", default=6)
    parser.add_argument("--seg_activation", default=None) #'sigmoid')
    parser.add_argument("--clsf_aux_params", default=dict(pooling='avg',             # one of 'avg', 'max'
                                                          dropout=None, #0.5         # dropout ratio, default is None
                                                          activation='sigmoid',      # activation function, default is None
                                                          classes=1,                 # define number of output labels
                                                        ))
    args = parser.parse_args()

    main(args)