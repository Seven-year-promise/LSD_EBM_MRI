
from __future__ import print_function, division
import os 
import random
from scipy import ndimage
import numpy as np
import scipy
from torch.utils.data import Dataset
import nibabel as nib



#%% Build the dataset 
class CSI_Dataset(Dataset):
    """xVertSeg Dataset"""
    
    def __init__(self, dataset_path, mode, offset=-2000):
        """
        Args:
            path_dataset(string): Root path to the whole dataset
            subset(string): 'train' or 'test' depend on which subset
        """
        self.idx = 1
        
        
        self.offset = offset
        self.mode=mode
        
        self.img_path = os.path.join(dataset_path,  'full')
        self.body_path = os.path.join(dataset_path, 'border')
                
        self.img_names =[f for f in os.listdir(self.img_path)]  # sanity test[f for f in os.listdir(self.img_path) if f in subset]


     
    def __len__(self):
        return len(self.img_names)
    
    
    def __getitem__(self, idx):
    
        img_name =  self.img_names[idx]
        body_name = self.img_names[idx].split('.')[0]+'_weight.nii.gz'

        img_file = os.path.join(self.img_path,  img_name)
        full_patch = np.round(nib.load(img_file).get_fdata()).astype('uint8')
        
        body_file = os.path.join(self.body_path,  body_name)
        body_patch = np.round(nib.load(body_file).get_fdata()).astype('uint8')
        
        full_patch, body_patch = extract_random_patch(full_patch, body_patch)
        
        
        full_patch = np.expand_dims(full_patch.astype('uint8'), axis=0)
        body_patch = np.expand_dims(body_patch.astype('uint8'), axis=0)

        self.idx+=1


        return  img_name, full_patch, body_patch

    
#%% Extract the 128*128*128 patch
def extract_random_patch(full_patch, body_patch, patch_size=128):

    pad = 30 # It has to account for both the minimum zoom and the maximum shift look to other datasets to do it properly
    
    full_patch = np.pad(full_patch, ((pad, pad), (pad, pad),(pad, pad) ), 'constant', constant_values=0)
    body_patch = np.pad(body_patch, ((pad, pad), (pad, pad),(pad, pad) ), 'constant', constant_values=0)
   
    zoom = 1+random.randint(-10,10)*0.01
    full_patch =  scipy.ndimage.zoom(full_patch, zoom, order=0) # Random zoom between 0.9 and 1.1
    body_patch =  scipy.ndimage.zoom(body_patch, zoom, order=0) # Random zoom between 0.9 and 1.1
        
    max_shift = 7 # note that this value and padding have to agree 
    shift_x = random.randint(-max_shift,max_shift)
    shift_y = random.randint(-max_shift,max_shift)
    shift_z = random.randint(-max_shift,max_shift)
    
    x,y,z=ndimage.measurements.center_of_mass(full_patch) # Shift
    
    x = x+shift_x 
    y = y+shift_y 
    z = z+shift_z 
    
    #extract the patch and padding
    x_low = int(round(x-patch_size/2))
    x_up =  int(round(x+patch_size/2))
    y_low = int(round(y-patch_size/2))
    y_up =  int(round(y+patch_size/2))
    z_low = int(round(z-patch_size/2))
    z_up =  int(round(z+patch_size/2))

    full_patch = full_patch[x_low:x_up, y_low:y_up ,z_low:z_up]
    body_patch = body_patch[x_low:x_up, y_low:y_up ,z_low:z_up]
     
    return full_patch, body_patch




