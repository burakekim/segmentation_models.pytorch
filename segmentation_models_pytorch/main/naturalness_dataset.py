
import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor

class MapInWild_Naturalness(torch.utils.data.Dataset): 
    
    BAND_SETS: Dict[str, Tuple[str, ...]] = {
        "all": (
            "VV",
            "VH",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
            "2020_Map",
            "avg_rad"), 

        "s1": ("VV", "VH"),
        "s2-all": (
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12"),
        "esa_wc": {"2020_Map"},
        "viirs":{"avg_rad"}
    }
    
    band_names = (
        "VV",
        "VH",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "2020_Map",
        "avg_rad"
    )
    
    RGB_BANDS = ["B4", "B3", "B2"]
    
    def __init__(
        self,
        split_file, 
        subsetpath = None, 
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        checksum: bool = False) -> None:
        """Initialize a new MapInWild dataset instance.
        """
        assert split in ["train","validation", "test"]

        self.band_indices = torch.tensor([self.band_names.index(b) for b in bands]).long()
        self.bands = bands

        self.root = root
        self.split = split
        self.subsetpath = subsetpath

        self.transforms = transforms
        self.checksum = checksum
        
        split_dataframe = pd.read_csv(split_file)
        self.ids = split_dataframe[split].dropna().values.tolist() 
        self.ids = [int(i) for i in self.ids]
        # self.ids = self.ids[:100] #######################################
        # print("Only 100 IDs")

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        
        filename = self.ids[index]
        
        mask = self._load_raster(filename, "NI")
        mask = mask / 100 
        mask[mask<0] = 0

        if self.subsetpath is None:
            s2 = self._load_raster(filename, "s2_summer")
        if self.subsetpath is not None:
            season = self.get_subset_s2_season(self.subsetpath, filename)
            s2 = self._load_raster(filename,str(season))
        s2 = s2/10000 

        s1 = self._load_raster(filename, "S1")
        s1_scaled = self.scale_range(input=s1.float(),min=0,max=1)

        esa_wc = self._load_raster(filename, "ESA_WC")
        esa_wc = self.scale_range(input=esa_wc.float(),min=0,max=1)
        
        viirs  = self._load_raster(filename, "VIIRS")
        viirs_scaled = self.scale_range(input=viirs.float(),min=0,max=1)
        
        image = torch.cat(tensors=[s1_scaled, s2, esa_wc, viirs_scaled], dim=0) 
        image = torch.index_select(image, dim=0, index=self.band_indices)
        
        if self.transforms is not None: 
            image = np.transpose(image.numpy(), (1, 2, 0))
            mask = np.transpose(mask.numpy(), (1, 2, 0))          

            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            image = np.transpose(image,(2, 0, 1))
            mask = np.transpose(mask,(2, 0, 1))

        return image, mask, filename 
    
    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_raster(self, filename: str, source: str) -> Tensor:
        """Load a single raster image or target.
        Args:
            filename: name of the file to load
            source: one of "mask", "s1", or "s2"
        Returns:
            the raster image or target
        """
        with rasterio.open(
                os.path.join(self.root,
                                "{}".format(source), ## source = the modality
                                "{}.tif".format(filename), ## the ids
                )
        ) as f:
            array = f.read().astype(np.int32)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor
        
    @staticmethod
    def scale_range(input, min, max):
        input += -(torch.min(input))
        input /= (1e-9 + torch.max(input) / (max - min + 1e-9))
        input += min
        return input
              
    @staticmethod
    def get_subset_s2_season(path_to_subset_seasons, data_id):
        """Get the subset season. 
        """
        subset_season_dataframe = pd.read_csv(path_to_subset_seasons)
        sts = subset_season_dataframe['single_temporal_subset'].tolist()
        im_id = subset_season_dataframe['imagePath'].tolist()
        zip_sts_id = dict(zip(im_id, sts))
        season = zip_sts_id[data_id]  
        s2_season = "s2_{}".format(season.lower())

        return s2_season
 