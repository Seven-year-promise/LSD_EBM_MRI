from __future__ import print_function, division
import os
import argparse
import time
import pickle
from torch.utils.data import DataLoader
import numpy as np
import scipy
import csv

import torch
import torch.optim as optim
import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc

from get_model_lsd_ebm import LSD_EBM_model
from get_model_lsd_ebm import args as lsd_ebm_args

from get_model_lebm import LEBM_model
from get_model_lebm import args as lebm_args

from get_model_vae import VAE_model
from get_model_vae import args as vae_args

from utils import calc_VolumetricSimilarity, calc_SimpleHausdorffDistance, calc_Sensitivity_Sets, calc_Specificity_Sets, calc_Hausdorff_Distance, calc_Normalized_Mutual_Information, calc_Cohen_kappa
import pandas as pd

def DiceCoeff(pred, gt):
     pred = pred.to('cpu').numpy()
     gt = gt.to('cpu').numpy()
     
     #if gt is all zero (use inverse to count)
     if np.count_nonzero(gt) == 0:
         gt = gt+1
         pred = 1-pred
         
     dice_score = dc(pred.astype(np.int),gt.astype(np.int)) 
     return dice_score

#%%Main
if  __name__ == "__main__" :   
   
    run_name = "reconstrcution_metrics"
    main_path="./comp_results/" + run_name + "/" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    with open(main_path + "hyper_para.txt", "w") as output:  ## creates new file but empty
        for arg in vars(lsd_ebm_args):
            print(arg, getattr(lsd_ebm_args, arg))
            output.write(str(arg) + "\t" + str(getattr(lsd_ebm_args, arg)) + "\n")

    #dice_recon_h_csv_file = open(on.path.join(main_path, 'dice_h_recon.csv'), 'w', newline='')
    #dice_recon_h_csv_writer = csv.writer(dice_recon_h_csv_file, delimiter=';',
    #                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    out_f = open(main_path + "output.txt",'w')

    lsd_ebm_model = LSD_EBM_model()
    lebm_model = LEBM_model()
    vae_model = VAE_model()

    lsd_dice_recon_l = {}
    lsd_dice_recon_h = {}
    lsd_vs_recon_l = {}  # volumetric similarity
    lsd_vs_recon_h = {}
    lsd_hd_recon_l = {} # Hausdordd distance
    lsd_hd_recon_h = {}
    lsd_sen_recon_l = {} # Sensitivity via Sets   
    lsd_sen_recon_h = {}
    lsd_spec_recon_l = {} # Specificity  
    lsd_spec_recon_h = {}
    lsd_nmi_recon_l = {} #   Normalized Mutual Information (NMI)
    lsd_nmi_recon_h = {}
    lsd_ck_recon_l = {} # Cohen’s kappa
    lsd_ck_recon_h = {}

    lebm_dice_recon_l = {}
    lebm_dice_recon_h = {}
    lebm_vs_recon_l = {} # volumetric similarity
    lebm_vs_recon_h = {}
    lebm_hd_recon_l = {} # Hausdordd distance
    lebm_hd_recon_h = {}
    lebm_sen_recon_l = {} # Sensitivity via Sets   
    lebm_sen_recon_h = {}
    lebm_spec_recon_l = {} # Specificity  
    lebm_spec_recon_h = {}
    lebm_nmi_recon_l = {} #   Normalized Mutual Information (NMI)
    lebm_nmi_recon_h = {}
    lebm_ck_recon_l = {} # Cohen’s kappa
    lebm_ck_recon_h = {}

    vae_dice_recon_l = {}
    vae_dice_recon_h = {}
    vae_vs_recon_l = {} # volumetric similarity
    vae_vs_recon_h = {}
    vae_hd_recon_l = {} # Hausdordd distance
    vae_hd_recon_h = {}
    vae_sen_recon_l = {} # Sensitivity via Sets   
    vae_sen_recon_h = {}
    vae_spec_recon_l = {} # Specificity  
    vae_spec_recon_h = {}
    vae_nmi_recon_l = {} #   Normalized Mutual Information (NMI)
    vae_nmi_recon_h = {}
    vae_ck_recon_l = {} # Cohen’s kappa
    vae_ck_recon_h = {}
    
    print("### Make predictions on test dataset")

    experiment_recon = 'recon4_augmheavy'
    experiment_recon_save = 'my_recon4_augmheavy'
    patients = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
    verts_recon = [1,2,3,4,5]
    testds_path = "./MRI_reconstructions/" 

    steps = [2, 15, 20]
    lsd_base_model_path = "./Vert_data/LSD-EBM_Vert_step_"
    lebm_base_model_path = "./Vert_data/LEBM_Vert_MCMC_step_"
    vae_base_model_path = "./Vert_data/VAE_Vert_01"

    for step in steps:
        print("LOAD LSD EBM MODEL EP ", lsd_base_model_path+str(step), lsd_ebm_args.load_from_ep, file=out_f)
        lsd_ebm_model.load_model(lsd_base_model_path+str(step))
        print("LOAD LEBM MODEL EP ", lebm_base_model_path+str(step), lebm_args.load_from_ep, file=out_f)
        lebm_model.load_model(lebm_base_model_path+str(step))
        if step == 2:
            print("LOAD VAE MODEL EP ", vae_base_model_path, vae_args.load_from_ep, file=out_f)
            vae_model.load_model(vae_base_model_path)

        lsd_dice_score_h = []
        lsd_dice_score_l = []
        lsd_vs_score_h = []
        lsd_vs_score_l = []
        lsd_hd_score_h = []
        lsd_hd_score_l = []
        lsd_sen_score_h = []
        lsd_sen_score_l = []
        lsd_spec_score_h = []
        lsd_spec_score_l = []
        lsd_nmi_score_h = []
        lsd_nmi_score_l = []
        lsd_ck_score_h = []
        lsd_ck_score_l = []

        lebm_dice_score_h = []
        lebm_dice_score_l = []
        lebm_vs_score_h = []
        lebm_vs_score_l = []
        lebm_hd_score_h = []
        lebm_hd_score_l = []
        lebm_sen_score_h = []
        lebm_sen_score_l = []
        lebm_spec_score_h = []
        lebm_spec_score_l = []
        lebm_nmi_score_h = []
        lebm_nmi_score_l = []
        lebm_ck_score_h = []
        lebm_ck_score_l = []

        if step == 2:
            vae_dice_score_h = []
            vae_dice_score_l = []
            vae_vs_score_h = []
            vae_vs_score_l = []
            vae_hd_score_h = []
            vae_hd_score_l = []
            vae_sen_score_h = []
            vae_sen_score_l = []
            vae_spec_score_h = []
            vae_spec_score_l = []
            vae_nmi_score_h = []
            vae_nmi_score_l = []
            vae_ck_score_h = []
            vae_ck_score_l = []

        for patient in patients:
            print("## Patient {}".format(patient), file=out_f)
            for idx in range(len(verts_recon)):
                print("# Vertebra {}".format(verts_recon[idx]), file=out_f)

                # load the high-quality data
                hq_path  = os.path.join(testds_path, experiment_recon,patient+'_mri',str(verts_recon[idx])+'_HQ.nii.gz')

                if not os.path.exists(hq_path):
                    print("file not found", file=out_f)
                    continue

                hq_nib = nib.load(hq_path)
                hq_patch = hq_nib.get_fdata().astype("int8")
                print(hq_patch.max(), file=out_f)
                hq_patch[hq_patch!=0]=1
                print(hq_patch.min(), file=out_f)
                print(hq_patch.max(), file=out_f)
                full_h_patch = torch.from_numpy(hq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

                # load the low-quality data
                lq_path  = os.path.join(testds_path, experiment_recon,patient+'_mri',str(verts_recon[idx])+'_LQ.nii.gz')
                lq_nib = nib.load(lq_path)
                lq_patch = lq_nib.get_fdata().astype("int8")
                print(lq_patch.max(), file=out_f)
                lq_patch[lq_patch!=0]=1
                print(lq_patch.min(), file=out_f)
                print(lq_patch.max(), file=out_f)
                full_l_patch = torch.from_numpy(lq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

                # predicted data lsd ebm
                l_pred_patch_lsd = lsd_ebm_model.reconstruction(full_l_patch, step=step)
                h_pred_patch_lsd = lsd_ebm_model.reconstruction(full_h_patch, step=step)

                

                l_reconstruct_dice_lsd = DiceCoeff(l_pred_patch_lsd, full_h_patch)
                h_reconstruct_dice_lsd = DiceCoeff(h_pred_patch_lsd, full_h_patch)
                l_reconstruct_vs_lsd = calc_VolumetricSimilarity(l_pred_patch_lsd, full_h_patch[0, 0, :, :, :], c=1) 
                h_reconstruct_vs_lsd = calc_VolumetricSimilarity(h_pred_patch_lsd, full_h_patch[0, 0, :, :, :], c=1) 
                l_reconstruct_hd_lsd = 0 #calc_Hausdorff_Distance(l_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                h_reconstruct_hd_lsd = 0 #calc_Hausdorff_Distance(h_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                l_reconstruct_sen_lsd = calc_Sensitivity_Sets(l_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                h_reconstruct_sen_lsd = calc_Sensitivity_Sets(h_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                l_reconstruct_spec_lsd = calc_Specificity_Sets(l_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                h_reconstruct_spec_lsd = calc_Specificity_Sets(h_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                l_reconstruct_nmi_lsd = calc_Normalized_Mutual_Information(l_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                h_reconstruct_nmi_lsd = calc_Normalized_Mutual_Information(h_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                l_reconstruct_ck_lsd = calc_Cohen_kappa(l_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                h_reconstruct_ck_lsd = calc_Cohen_kappa(h_pred_patch_lsd[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 


                # predicted data lebm
                l_pred_patch_lebm = lebm_model.reconstruction(full_l_patch)
                h_pred_patch_lebm = lebm_model.reconstruction(full_h_patch)

                l_reconstruct_dice_lebm = DiceCoeff(l_pred_patch_lebm, full_h_patch)
                h_reconstruct_dice_lebm = DiceCoeff(h_pred_patch_lebm, full_h_patch)
                l_reconstruct_vs_lebm = calc_VolumetricSimilarity(l_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) # volumetric similarity
                h_reconstruct_vs_lebm = calc_VolumetricSimilarity(h_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) # volumetric similarity
                l_reconstruct_hd_lebm = 0 # calc_SimpleHausdorffDistance(l_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                h_reconstruct_hd_lebm = 0 # calc_SimpleHausdorffDistance(h_pred_patch_lebm[0, :, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                l_reconstruct_sen_lebm = calc_Sensitivity_Sets(l_pred_patch_lebm[0, 0 :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                h_reconstruct_sen_lebm = calc_Sensitivity_Sets(h_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                l_reconstruct_spec_lebm = calc_Specificity_Sets(l_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                h_reconstruct_spec_lebm = calc_Specificity_Sets(h_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                l_reconstruct_nmi_lebm= calc_Normalized_Mutual_Information(l_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                h_reconstruct_nmi_lebm = calc_Normalized_Mutual_Information(h_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                l_reconstruct_ck_lebm = calc_Cohen_kappa(l_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                h_reconstruct_ck_lebm = calc_Cohen_kappa(h_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 

                # predition VAE 
                if step ==2:
                    l_pred_patch_vae = vae_model.reconstruction(full_l_patch)
                    h_pred_patch_vae = vae_model.reconstruction(full_h_patch)

                    l_reconstruct_dice_vae = DiceCoeff(l_pred_patch_vae, full_h_patch)
                    h_reconstruct_dice_vae = DiceCoeff(h_pred_patch_vae, full_h_patch)
                    l_reconstruct_vs_vae = calc_VolumetricSimilarity(l_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) # volumetric similarity
                    h_reconstruct_vs_vae = calc_VolumetricSimilarity(h_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) # volumetric similarity
                    l_reconstruct_hd_vae = 0 # calc_SimpleHausdorffDistance(l_pred_patch_lebm[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                    h_reconstruct_hd_vae = 0 # calc_SimpleHausdorffDistance(h_pred_patch_lebm[0, :, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                    l_reconstruct_sen_vae = calc_Sensitivity_Sets(l_pred_patch_vae[0, 0 :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                    h_reconstruct_sen_vae = calc_Sensitivity_Sets(h_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                    l_reconstruct_spec_vae = calc_Specificity_Sets(l_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                    h_reconstruct_spec_vae = calc_Specificity_Sets(h_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :], c=1) 
                    l_reconstruct_nmi_vae= calc_Normalized_Mutual_Information(l_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                    h_reconstruct_nmi_vae = calc_Normalized_Mutual_Information(h_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                    l_reconstruct_ck_vae = calc_Cohen_kappa(l_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 
                    h_reconstruct_ck_vae = calc_Cohen_kappa(h_pred_patch_vae[0, 0, :, :, :], full_h_patch[0, 0, :, :, :]) 


                out_path = os.path.join(main_path, experiment_recon_save, patient+'_mri')
                if not os.path.exists(out_path):
                    print('Creating directory at:', out_path, file=out_f)
                    os.makedirs(out_path)

                l_out_path_lsd = os.path.join(out_path, str(verts_recon[idx])+'_step'+str(step) +'_LQ_LSD_EBM.nii.gz')
                h_out_path_lsd = os.path.join(out_path, str(verts_recon[idx])+'_step'+str(step) +'_HQ_LSD_EBM.nii.gz')

                l_out_path_lebm = os.path.join(out_path, str(verts_recon[idx])+'_step'+str(step) +'_LQ_LEBM.nii.gz')
                h_out_path_lebm = os.path.join(out_path, str(verts_recon[idx])+'_step'+str(step) +'_HQ_LEBM.nii.gz')

                nib.save(nib.Nifti1Image(np.squeeze(l_pred_patch_lsd.to('cpu').numpy()[0]),affine=None), l_out_path_lsd)
                nib.save(nib.Nifti1Image(np.squeeze(h_pred_patch_lsd.to('cpu').numpy()[0]),affine=None), h_out_path_lsd)
                
                nib.save(nib.Nifti1Image(np.squeeze(l_pred_patch_lebm.to('cpu').numpy()[0]),affine=None), l_out_path_lebm)
                nib.save(nib.Nifti1Image(np.squeeze(h_pred_patch_lebm.to('cpu').numpy()[0]),affine=None), h_out_path_lebm)

                if step ==2:
                    l_out_path_vae = os.path.join(out_path, str(verts_recon[idx]) +'_LQ_VAE.nii.gz')
                    h_out_path_vae = os.path.join(out_path, str(verts_recon[idx]) +'_HQ_VAE.nii.gz')
                    nib.save(nib.Nifti1Image(np.squeeze(l_pred_patch_vae.to('cpu').numpy()[0]),affine=None), l_out_path_vae)
                    nib.save(nib.Nifti1Image(np.squeeze(h_pred_patch_vae.to('cpu').numpy()[0]),affine=None), h_out_path_vae)

                del l_pred_patch_lsd
                del h_pred_patch_lsd
                del l_pred_patch_lebm
                del h_pred_patch_lebm
                if step == 2:
                    del l_pred_patch_vae
                    del h_pred_patch_vae

                lsd_dice_score_l.append(l_reconstruct_dice_lsd)
                lsd_dice_score_h.append(h_reconstruct_dice_lsd)
                lsd_vs_score_l.append(l_reconstruct_vs_lsd)
                lsd_vs_score_h.append(h_reconstruct_vs_lsd)
                lsd_hd_score_l.append(l_reconstruct_hd_lsd)
                lsd_hd_score_h.append(h_reconstruct_hd_lsd)
                lsd_sen_score_l.append(l_reconstruct_sen_lsd)
                lsd_sen_score_h.append(h_reconstruct_sen_lsd)
                lsd_spec_score_l.append(l_reconstruct_spec_lsd)
                lsd_spec_score_h.append(h_reconstruct_spec_lsd)
                lsd_nmi_score_l.append(l_reconstruct_nmi_lsd)
                lsd_nmi_score_h.append(h_reconstruct_nmi_lsd)
                lsd_ck_score_l.append(l_reconstruct_ck_lsd)
                lsd_ck_score_h.append(h_reconstruct_ck_lsd)

                lebm_dice_score_l.append(l_reconstruct_dice_lebm)
                lebm_dice_score_h.append(h_reconstruct_dice_lebm)
                lebm_vs_score_l.append(l_reconstruct_vs_lebm)
                lebm_vs_score_h.append(h_reconstruct_vs_lebm)
                lebm_hd_score_l.append(l_reconstruct_hd_lebm)
                lebm_hd_score_h.append(h_reconstruct_hd_lebm)
                lebm_sen_score_l.append(l_reconstruct_sen_lebm)
                lebm_sen_score_h.append(h_reconstruct_sen_lebm)
                lebm_spec_score_l.append(l_reconstruct_spec_lebm)
                lebm_spec_score_h.append(h_reconstruct_spec_lebm)
                lebm_nmi_score_l.append(l_reconstruct_nmi_lebm)
                lebm_nmi_score_h.append(h_reconstruct_nmi_lebm)
                lebm_ck_score_l.append(l_reconstruct_ck_lebm)
                lebm_ck_score_h.append(h_reconstruct_ck_lebm)

                if step == 2:
                    vae_dice_score_l.append(l_reconstruct_dice_vae)
                    vae_dice_score_h.append(h_reconstruct_dice_vae)
                    vae_vs_score_l.append(l_reconstruct_vs_vae)
                    vae_vs_score_h.append(h_reconstruct_vs_vae)
                    vae_hd_score_l.append(l_reconstruct_hd_vae)
                    vae_hd_score_h.append(h_reconstruct_hd_vae)
                    vae_sen_score_l.append(l_reconstruct_sen_vae)
                    vae_sen_score_h.append(h_reconstruct_sen_vae)
                    vae_spec_score_l.append(l_reconstruct_spec_vae)
                    vae_spec_score_h.append(h_reconstruct_spec_vae)
                    vae_nmi_score_l.append(l_reconstruct_nmi_vae)
                    vae_nmi_score_h.append(h_reconstruct_nmi_vae)
                    vae_ck_score_l.append(l_reconstruct_ck_vae)
                    vae_ck_score_h.append(h_reconstruct_ck_vae)
                
            
            

        lsd_dice_recon_l[step] = lsd_dice_score_l
        lsd_dice_recon_l_pd = pd.DataFrame(lsd_dice_recon_l)
        lsd_dice_recon_h[step] = lsd_dice_score_h
        lsd_dice_recon_h_pd = pd.DataFrame(lsd_dice_recon_h)
        lsd_vs_recon_l[step] = lsd_vs_score_l
        lsd_vs_recon_l_pd = pd.DataFrame(lsd_vs_recon_l)
        lsd_vs_recon_h[step] = lsd_vs_score_h
        lsd_vs_recon_h_pd = pd.DataFrame(lsd_vs_recon_h)
        lsd_hd_recon_l[step] = lsd_hd_score_l
        lsd_hd_recon_l_pd = pd.DataFrame(lsd_hd_recon_l)
        lsd_hd_recon_h[step] = lsd_hd_score_h
        lsd_hd_recon_h_pd = pd.DataFrame(lsd_hd_recon_h)
        lsd_sen_recon_l[step] = lsd_sen_score_l
        lsd_sen_recon_l_pd = pd.DataFrame(lsd_sen_recon_l)
        lsd_sen_recon_h[step] = lsd_sen_score_h
        lsd_sen_recon_h_pd = pd.DataFrame(lsd_sen_recon_h)
        lsd_spec_recon_l[step] = lsd_spec_score_l
        lsd_spec_recon_l_pd = pd.DataFrame(lsd_spec_recon_l)
        lsd_spec_recon_h[step] = lsd_spec_score_h
        lsd_spec_recon_h_pd = pd.DataFrame(lsd_spec_recon_h)

        lsd_nmi_recon_l[step] = lsd_nmi_score_l
        lsd_nmi_recon_l_pd = pd.DataFrame(lsd_nmi_recon_l)
        lsd_nmi_recon_h[step] = lsd_nmi_score_h
        lsd_nmi_recon_h_pd = pd.DataFrame(lsd_nmi_recon_h)
        lsd_ck_recon_l[step] = lsd_ck_score_l
        lsd_ck_recon_l_pd = pd.DataFrame(lsd_ck_recon_l)
        lsd_ck_recon_h[step] = lsd_ck_score_h
        lsd_ck_recon_h_pd = pd.DataFrame(lsd_ck_recon_h)

        lebm_dice_recon_l[step] = lebm_dice_score_l
        lebm_dice_recon_l_pd = pd.DataFrame(lebm_dice_recon_l)
        lebm_dice_recon_h[step] = lebm_dice_score_h
        lebm_dice_recon_h_pd = pd.DataFrame(lebm_dice_recon_h)
        lebm_vs_recon_l[step] = lebm_vs_score_l
        lebm_vs_recon_l_pd = pd.DataFrame(lebm_vs_recon_l)
        lebm_vs_recon_h[step] = lebm_vs_score_h
        lebm_vs_recon_h_pd = pd.DataFrame(lebm_vs_recon_h)
        lebm_hd_recon_l[step] = lebm_hd_score_l
        lebm_hd_recon_l_pd = pd.DataFrame(lebm_hd_recon_l)
        lebm_hd_recon_h[step] = lebm_hd_score_h
        lebm_hd_recon_h_pd = pd.DataFrame(lebm_hd_recon_h)
        lebm_sen_recon_l[step] = lebm_sen_score_l
        lebm_sen_recon_l_pd = pd.DataFrame(lebm_sen_recon_l)
        lebm_sen_recon_h[step] = lebm_sen_score_h
        lebm_sen_recon_h_pd = pd.DataFrame(lebm_sen_recon_h)
        lebm_spec_recon_l[step] = lebm_spec_score_l
        lebm_spec_recon_l_pd = pd.DataFrame(lebm_spec_recon_l)
        lebm_spec_recon_h[step] = lebm_spec_score_h
        lebm_spec_recon_h_pd = pd.DataFrame(lebm_spec_recon_h)

        lebm_nmi_recon_l[step] = lebm_nmi_score_l
        lebm_nmi_recon_l_pd = pd.DataFrame(lebm_nmi_recon_l)
        lebm_nmi_recon_h[step] = lebm_nmi_score_h
        lebm_nmi_recon_h_pd = pd.DataFrame(lebm_nmi_recon_h)
        lebm_ck_recon_l[step] = lebm_ck_score_l
        lebm_ck_recon_l_pd = pd.DataFrame(lebm_ck_recon_l)
        lebm_ck_recon_h[step] = lebm_ck_score_h
        lebm_ck_recon_h_pd = pd.DataFrame(lebm_ck_recon_h)

        if step == 2:
            vae_dice_recon_l[step] = vae_dice_score_l
            vae_dice_recon_l_pd = pd.DataFrame(vae_dice_recon_l)
            vae_dice_recon_h[step] = vae_dice_score_h
            vae_dice_recon_h_pd = pd.DataFrame(vae_dice_recon_h)
            vae_vs_recon_l[step] = vae_vs_score_l
            vae_vs_recon_l_pd = pd.DataFrame(vae_vs_recon_l)
            vae_vs_recon_h[step] = vae_vs_score_h
            vae_vs_recon_h_pd = pd.DataFrame(vae_vs_recon_h)
            vae_hd_recon_l[step] = vae_hd_score_l
            vae_hd_recon_l_pd = pd.DataFrame(vae_hd_recon_l)
            vae_hd_recon_h[step] = vae_hd_score_h
            vae_hd_recon_h_pd = pd.DataFrame(vae_hd_recon_h)
            vae_sen_recon_l[step] = vae_sen_score_l
            vae_sen_recon_l_pd = pd.DataFrame(vae_sen_recon_l)
            vae_sen_recon_h[step] = vae_sen_score_h
            vae_sen_recon_h_pd = pd.DataFrame(vae_sen_recon_h)
            vae_spec_recon_l[step] = vae_spec_score_l
            vae_spec_recon_l_pd = pd.DataFrame(vae_spec_recon_l)
            vae_spec_recon_h[step] = vae_spec_score_h
            vae_spec_recon_h_pd = pd.DataFrame(vae_spec_recon_h)

            vae_nmi_recon_l[step] = vae_nmi_score_l
            vae_nmi_recon_l_pd = pd.DataFrame(vae_nmi_recon_l)
            vae_nmi_recon_h[step] = vae_nmi_score_h
            vae_nmi_recon_h_pd = pd.DataFrame(vae_nmi_recon_h)
            vae_ck_recon_l[step] = vae_ck_score_l
            vae_ck_recon_l_pd = pd.DataFrame(vae_ck_recon_l)
            vae_ck_recon_h[step] = vae_ck_score_h
            vae_ck_recon_h_pd = pd.DataFrame(vae_ck_recon_h)


    
    #lsd ebm
    lsd_dice_recon_l_pd.to_csv(os.path.join(main_path, 'lsd_ebm_dice_l_recon.csv'), sep='\t', index=False)
    lsd_dice_recon_h_pd.to_csv(os.path.join(main_path, 'lsd_ebm_dice_h_recon.csv'), sep='\t', index=False)
    lsd_vs_recon_l_pd.to_csv(os.path.join(main_path, 'lsd_ebm_vs_l_recon.csv'), sep='\t', index=False)
    lsd_vs_recon_h_pd.to_csv(os.path.join(main_path, 'lsd_ebm_vs_h_recon.csv'), sep='\t', index=False)
    lsd_hd_recon_l_pd.to_csv(os.path.join(main_path, 'lsd_ebm_hd_l_recon.csv'), sep='\t', index=False)
    lsd_hd_recon_h_pd.to_csv(os.path.join(main_path, 'lsd_ebm_hd_h_recon.csv'), sep='\t', index=False)
    lsd_sen_recon_l_pd.to_csv(os.path.join(main_path, 'lsd_ebm_sen_l_recon.csv'), sep='\t', index=False)
    lsd_sen_recon_h_pd.to_csv(os.path.join(main_path, 'lsd_ebm_sen_h_recon.csv'), sep='\t', index=False)
    lsd_spec_recon_l_pd.to_csv(os.path.join(main_path, 'lsd_ebm_spec_l_recon.csv'), sep='\t', index=False)
    lsd_spec_recon_h_pd.to_csv(os.path.join(main_path, 'lsd_ebm_spec_h_recon.csv'), sep='\t', index=False)

    lsd_nmi_recon_l_pd.to_csv(os.path.join(main_path, 'lsd_ebm_nmi_l_recon.csv'), sep='\t', index=False)
    lsd_nmi_recon_h_pd.to_csv(os.path.join(main_path, 'lsd_ebm_nmi_h_recon.csv'), sep='\t', index=False)
    lsd_ck_recon_l_pd.to_csv(os.path.join(main_path, 'lsd_ebm_ck_l_recon.csv'), sep='\t', index=False)
    lsd_ck_recon_h_pd.to_csv(os.path.join(main_path, 'lsd_ebm_ck_h_recon.csv'), sep='\t', index=False)

    #lebm
    lebm_dice_recon_l_pd.to_csv(os.path.join(main_path, 'lebm_dice_l_recon.csv'), sep='\t', index=False)
    lebm_dice_recon_h_pd.to_csv(os.path.join(main_path, 'lebm_dice_h_recon.csv'), sep='\t', index=False)
    lebm_vs_recon_l_pd.to_csv(os.path.join(main_path, 'lebm_vs_l_recon.csv'), sep='\t', index=False)
    lebm_vs_recon_h_pd.to_csv(os.path.join(main_path, 'lebm_vs_h_recon.csv'), sep='\t', index=False)
    lebm_hd_recon_l_pd.to_csv(os.path.join(main_path, 'lebm_hd_l_recon.csv'), sep='\t', index=False)
    lebm_hd_recon_h_pd.to_csv(os.path.join(main_path, 'lebm_hd_h_recon.csv'), sep='\t', index=False)
    lebm_sen_recon_l_pd.to_csv(os.path.join(main_path, 'lebm_sen_l_recon.csv'), sep='\t', index=False)
    lebm_sen_recon_h_pd.to_csv(os.path.join(main_path, 'lebm_sen_h_recon.csv'), sep='\t', index=False)
    lebm_spec_recon_l_pd.to_csv(os.path.join(main_path, 'lebm_spec_l_recon.csv'), sep='\t', index=False)
    lebm_spec_recon_h_pd.to_csv(os.path.join(main_path, 'lebm_spec_h_recon.csv'), sep='\t', index=False)

    lebm_nmi_recon_l_pd.to_csv(os.path.join(main_path, 'lebm_nmi_l_recon.csv'), sep='\t', index=False)
    lebm_nmi_recon_h_pd.to_csv(os.path.join(main_path, 'lebm_nmi_h_recon.csv'), sep='\t', index=False)
    lebm_ck_recon_l_pd.to_csv(os.path.join(main_path, 'lebm_ck_l_recon.csv'), sep='\t', index=False)
    lebm_ck_recon_h_pd.to_csv(os.path.join(main_path, 'lebm_ck_h_recon.csv'), sep='\t', index=False)

    #vae
    vae_dice_recon_l_pd.to_csv(os.path.join(main_path, 'vae_dice_l_recon.csv'), sep='\t', index=False)
    vae_dice_recon_h_pd.to_csv(os.path.join(main_path, 'vae_dice_h_recon.csv'), sep='\t', index=False)
    vae_vs_recon_l_pd.to_csv(os.path.join(main_path, 'vae_vs_l_recon.csv'), sep='\t', index=False)
    vae_vs_recon_h_pd.to_csv(os.path.join(main_path, 'vae_vs_h_recon.csv'), sep='\t', index=False)
    vae_hd_recon_l_pd.to_csv(os.path.join(main_path, 'vae_hd_l_recon.csv'), sep='\t', index=False)
    vae_hd_recon_h_pd.to_csv(os.path.join(main_path, 'vae_hd_h_recon.csv'), sep='\t', index=False)
    vae_sen_recon_l_pd.to_csv(os.path.join(main_path, 'vae_sen_l_recon.csv'), sep='\t', index=False)
    vae_sen_recon_h_pd.to_csv(os.path.join(main_path, 'vae_sen_h_recon.csv'), sep='\t', index=False)
    vae_spec_recon_l_pd.to_csv(os.path.join(main_path, 'vae_spec_l_recon.csv'), sep='\t', index=False)
    vae_spec_recon_h_pd.to_csv(os.path.join(main_path, 'vae_spec_h_recon.csv'), sep='\t', index=False)

    vae_nmi_recon_l_pd.to_csv(os.path.join(main_path, 'vae_nmi_l_recon.csv'), sep='\t', index=False)
    vae_nmi_recon_h_pd.to_csv(os.path.join(main_path, 'vae_nmi_h_recon.csv'), sep='\t', index=False)
    vae_ck_recon_l_pd.to_csv(os.path.join(main_path, 'vae_ck_l_recon.csv'), sep='\t', index=False)
    vae_ck_recon_h_pd.to_csv(os.path.join(main_path, 'vae_ck_h_recon.csv'), sep='\t', index=False)

    """
    with open(os.path.join(main_path, 'lsd_ebm_dice_l_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = lsd_dice_recon_l.keys())
        #for key, value in lsd_dice_recon_l.items():
        #    csv_writer.writerow([key, value])
        csv_writer.writeheader()
        csv_writer.writerow(lsd_dice_recon_l)
    
    with open(os.path.join(main_path, 'lsd_ebm_dice_h_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = steps)
        csv_writer.writeheader()
        csv_writer.writerows(lsd_dice_recon_h)

    with open(os.path.join(main_path, 'lsd_ebm_vs_l_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = steps)
        csv_writer.writeheader()
        csv_writer.writerows(lsd_dice_recon_l)
    
    with open(os.path.join(main_path, 'lsd_ebm_vs_h_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = steps)
        csv_writer.writeheader()
        csv_writer.writerows(lsd_vs_recon_h)

    # lebm
    with open(os.path.join(main_path, 'lebm_dice_l_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = steps)
        csv_writer.writeheader()
        csv_writer.writerows(lebm_dice_recon_l)
    
    with open(os.path.join(main_path, 'lebm_dice_h_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = steps)
        csv_writer.writeheader()
        csv_writer.writerows(lebm_dice_recon_h)

    with open(os.path.join(main_path, 'lebm_vs_l_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = steps)
        csv_writer.writeheader()
        csv_writer.writerows(lebm_dice_recon_l)
    
    with open(os.path.join(main_path, 'lebm_vs_h_recon.csv'), 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames = steps)
        csv_writer.writeheader()
        csv_writer.writerows(lebm_vs_recon_h)

    """

