import torch
import numpy as np
from models.model.unet import UNet


###########################################################################################
###########################################################################################

WC_COLOR_MAP = {
    0: [0,0,0],
    10: [0, 100,0],
    20: [255,187,34],
    30: [255,255,76],
    40: [240,150,255],
    50: [250,0,0],
    60: [180,180,180],
    70: [240,240,240],
    80: [0,100,200],
    90: [0,150,160],
    95: [0,207,117],
    100: [250,230,160]
}



def normalize_image(image):
    
    image = image.astype(np.float32)
    normalized_image = np.zeros_like(image, dtype=np.float32)
    valid_mask = np.ones(image.shape[1:], dtype=bool)
    
    for band_idx in range(image.shape[0]):
        band = image[band_idx]

        #Handle NaN values
        valid_mask &= ~np.isnan(band)
        
        band = np.nan_to_num(band, nan=0.0)
      
        mean = np.mean(band)
        std = np.std(band)

        #Check for null stds
        std = max(std, 1e-3)

        normalized_band = (band - mean) / std
        normalized_band = np.clip(normalized_band, -10, 10)
        
        normalized_image[band_idx] = normalized_band

    valid_mask = valid_mask.astype(np.float32)
    normalized_image = np.concatenate([normalized_image, valid_mask[None,...]], axis=0)
    return normalized_image


def remap_mask(mapped, reverse_map):
    mask = np.zeros_like(mapped)
    for old_id, new_id in reverse_map.items():
        mask[mapped == old_id] = new_id
    return mask


###########################################################################################
###########################################################################################



class Model():
    def __init__(self, checkpoint_path, model_name = None, device = "cpu"):

         # Setup Basics
        self.device = device
        self.mask_tag = {}
        self.checkpoint_path = checkpoint_path
        self.mask_tag['model_used'] = self.checkpoint_path
        self.model_name = model_name

        # Load Checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location = self.device)

        #Characteristics from the checkpoint metadata
        if (self.checkpoint['in_channels'] and self.checkpoint['out_channels']):
            self.in_channels = self.checkpoint['in_channels']
            self.out_channels = self.checkpoint['out_channels']
        else: raise Exception("Checkpoint is missing in_channels and out_channels metadata")

        if self.checkpoint['bands']: 
            self.bands = self.checkpoint['bands']
            self.mask_tag['bands_used'] = self.bands
        else: raise Exception("Checkpoint metadata is missing band specifications")

        if self.checkpoint['NanChannel']: self.Nan_Channel = self.checkpoint['NanChannel']
        else: self.Nan_Channel = False

        if self.checkpoint['label_map']: 
            self.label_map = self.checkpoint['label_map']
            self.mask_tag['label_map'] = self.label_map
        if self.checkpoint['wc_code_map']: 
            self.wc_code_map = self.checkpoint['wc_code_map']
            self.mask_tag['wc_code_map'] = self.wc_code_map
            color_map = {}
            for _, val in self.wc_code_map.items(): 
                color_map[val] = WC_COLOR_MAP[val]
            self.color_map = color_map
            self.mask_tag['color_map'] = color_map



        # Setup the Model
        self.model = UNet(in_channels=self.in_channels, out_channels=self.out_channels)
        self.model.load_state_dict(self.checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()





    """
    Inference Method
    INPUT:
        Expects a numpy array of size (BxHxW)
        Height and Width must be a multiple of 32
    OUTPUT:
        Returns a numpy array of size (HxW) based on input
    """
    @torch.no_grad()
    def inference(self, image):
        # Safety Checks
        input_channels, height, width = image.shape
        exptected_channels = self.in_channels
        if self.Nan_Channel: exptected_channels -= 1
        if input_channels != exptected_channels:
            print(f"Incorrect input image shape, expected {exptected_channels} input channels, received {input_channels}")
            return None
        elif (height % 32 != 0) or (width % 32 != 0):
            print(f"Height and Width must be multiples of 32 - given {height}x{width}")
            return None
        else:
            try:
                # Setup the image
                image = normalize_image(image)
                image = torch.from_numpy(image)
                image = image.unsqueeze(0)
                image = image.to(self.device)

                # Inference
                pred = self.model(image)

                # Prep the output
                pred = pred.squeeze()
                mask = torch.argmax(pred, dim=0)

                # Adapt mask to hold the WC labels
                if self.wc_code_map:
                    for key, val in self.wc_code_map.items():
                        mask[mask == key] = val

                return mask

            except Exception as e: print(f'Correct input given, but error completing the inference \n Exception: {e}')



