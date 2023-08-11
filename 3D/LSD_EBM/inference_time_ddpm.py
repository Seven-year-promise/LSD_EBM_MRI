from __future__ import print_function, division
import os
import argparse
import time
import pickle
from torch.utils.data import DataLoader
import numpy as np
import scipy
import csv
from time import process_time

import torch
import torch.optim as optim
import nibabel as nib
import torch.nn as nn


from get_model_ddpm import DDPM_model
import get_model_ddpm
import pandas as pd


#%%Main
if  __name__ == "__main__" :   
   
    main_path="./comp_results/" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    #dice_recon_h_csv_file = open(on.path.join(main_path, 'dice_h_recon.csv'), 'w', newline='')
    #dice_recon_h_csv_writer = csv.writer(dice_recon_h_csv_file, delimiter=';',
    #                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    out_f = open(main_path + "processing_time.txt",'w')

    ddpm_model = DDPM_model()

    ddpm_base_model_path = "./DDPM_Vert/logs/DDPM_Vert"

    print("LOAD DDPM", ddpm_base_model_path, file=out_f)
    ddpm_model.load_model(ddpm_base_model_path)

    testds_path = "./MRI_reconstructions/" 

    experiment_recon = 'recon4_augmheavy'

    patient = '17'
    print("## Patient {}".format(patient), file=out_f)

    hq_path  = os.path.join(testds_path, experiment_recon, patient+'_mri', '2_HQ.nii.gz')


    hq_nib = nib.load(hq_path)
    hq_patch = hq_nib.get_fdata().astype("int8")
    print(hq_patch.max(), file=out_f)
    hq_patch[hq_patch!=0]=1
    print(hq_patch.min(), file=out_f)
    print(hq_patch.max(), file=out_f)
    full_h_patch = torch.from_numpy(hq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

    t3 = process_time() 

    h_pred_patch_ddpm = ddpm_model.generation(full_h_patch)

    t4 = process_time() 

    print("step ", step, "LSD-EBM processing time (s)", t2-t1, file=out_f)
    print("step ", step, "LEBM processing time (s)", t3-t2, file=out_f)
    print("T ", 1000, "DDPM processing time (s)", t4-t3, file=out_f)



