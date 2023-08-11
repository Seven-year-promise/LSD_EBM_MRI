import os
import numpy as np
import nibabel as nib
from medpy.metric.binary import dc
from scipy import ndimage
import argparse


def allign_recon(recon, gt, vert_id, max_shift_x = [-14,15], max_shift_y = [-10,17], max_shift_z = [-14,14], max_rot_xy = [-7,7], max_rot_yz = [-7,7], max_rot_xz = [-7,8]):
#def allign_recon(recon, gt, vert_id, max_shift_x = [-2,2], max_shift_y = [-4,-1], max_shift_z = [1,4], max_rot_xy = [-3,0], max_rot_yz = [-7,-3], max_rot_xz = [-7,-3]):

    recon_bin = recon.copy()
    gt_bin = gt.copy()

    recon_bin[recon_bin>0] = 1
    gt_bin[gt_bin>0] = 1
    
    best_dice = 0
    best_x = 0
    best_y = 0
    best_z = 0
    
    best_xy = 0
    best_yz = 0
    best_xz = 0
    flag = False
    initial_dice = dc(recon_bin,gt_bin)
    #print("shift")
    for x in range(max_shift_x[0],max_shift_x[1]):
        for y in range(max_shift_y[0],max_shift_y[1]):
            for z in range(max_shift_z[0],max_shift_z[1]):
                shifted_patch = np.roll(recon_bin.copy(), (x,y,z), axis=(0,1,2))
                dice = dc(shifted_patch,gt_bin) 
                #print("Vert {}    x: {}  y: {}  z: {}\tdice: {}".format(vert_id,x,y,z,dice))
                if dice>best_dice:
                    best_dice=dice
                    best_x = x
                    best_y = y
                    best_z = z
                    
    
    recon_bin = np.roll(recon_bin, (best_x,best_y,best_z), axis=(0,1,2))
    #print("rotation")
    for xy in range(max_rot_xy[0],max_rot_xy[1]):
        for yz in range(max_rot_yz[0],max_rot_yz[1]):
            for xz in range(max_rot_xz[0],max_rot_xz[1]):
                rolled_patch = ndimage.rotate(recon_bin.copy(), xy, axes = [0,1], reshape = False, order=0)
                rolled_patch = ndimage.rotate(rolled_patch, yz, axes = [1,2], reshape = False, order=0)
                rolled_patch = ndimage.rotate(rolled_patch, xz, axes = [0,2], reshape = False, order=0)
    
                dice = dc(rolled_patch,gt_bin) 
                #print("Vert {}    xy: {}  yz: {}  xz: {}\tdice: {}".format(vert_id,xy,yz,xz,dice))
                if dice>best_dice:
                    best_dice=dice
                    best_xy = xy
                    best_yz = yz
                    best_xz = xz
    
    #print("\nInitial Dice: {}\n".format(initial_dice))
    print("Best Dice at x: {}  y: {}  z: {}  xy: {}  yz: {}  xz: {}\tdice: {:.2f}".format(best_x,best_y,best_z,best_xy,best_yz,best_xz,best_dice*100))
    
    
    if (best_x==max_shift_x[0] or best_x==max_shift_x[1]-1) or (best_y==max_shift_y[0] or best_y==max_shift_y[1]-1) or (best_z==max_shift_z[0] or best_z==max_shift_z[1]-1) or (best_xy == max_rot_xy[0] or best_xy == max_rot_xy[1]-1) or (best_yz == max_rot_yz[0] or best_yz == max_rot_yz[1]-1) or (best_xz == max_rot_xz[0] or best_xz == max_rot_xz[1]-1):
        print("NOTE: there might be room of improvement as the best shift or rotation was the maximum allowed\n")
        flag = True
    
    recon = np.roll(recon, (best_x,best_y,best_z), axis=(0,1,2))
    
    recon = ndimage.rotate(recon, best_xy, axes = [0,1], reshape = False, order=0)
    recon = ndimage.rotate(recon, best_yz, axes = [1,2], reshape = False, order=0)
    recon = ndimage.rotate(recon, best_xz, axes = [0,2], reshape = False, order=0)
                           
    return recon, best_dice, flag




def extract_patch(patch, patch_size):
    x,y,z=ndimage.measurements.center_of_mass(patch)

    x_low = int(x-patch_size/2)
    x_up =  int(x+patch_size/2)
    y_low = int(y-patch_size/2)
    y_up =  int(y+patch_size/2)
    z_low = int(z-patch_size/2)
    z_up =  int(z+patch_size/2)
    
    return patch[x_low:x_up, y_low:y_up ,z_low:z_up], [x_low,x_up,y_low,y_up,z_low,z_up]
    

####################################################################################################################################################################################
parser = argparse.ArgumentParser(description='Reconstruction of MRI images')
parser.add_argument('--subject', default="01", help='choose which subject to perform reconstruction on')   
args = parser.parse_args()


patients = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']

for patient in patients:
    
    #experiment_recon = 'reconGan1_gen'
    experiment_recon = 'my_recon4_augmheavy'
    recon_quality = '_LSD-EBM_Vert_24'
    
    comparison = 2 # 1: recon with ctseg, 2: recon with gt, 3: ctseg with gt
    
    verts_recon = [1,2,3,4,5]
    verts_ct = [1,2,3,4,5]
    verts_gt = [1,2,3,4,5]
    
    
    experiment_CTseg = 'last_verse1_05FPscale'
    
    gt_dataset = 'Ground_Truth_original'
    gt_dataset = 'My_Ground_Truth'
    

    assert len(verts_recon)==len(verts_gt), "Different number of recon and gt vertebrae"
    
    print("Experiment: {}".format(experiment_recon))
    print("Subject: {}".format(patient))
    print("Recon quality: {}".format(recon_quality))
    print("Comparison: {}".format(comparison))
    
    main_path=os.path.dirname(os.path.realpath(__file__))
    
    if comparison == 3:
        recon_path = os.path.join(main_path, "CT_segmentations",experiment_CTseg,patient+'_ct',patient+'_ct_seg.nii.gz')
        
    if comparison == 1:
        gt_path = os.path.join(main_path, "CT_segmentations",experiment_CTseg,patient+'_ct',patient+'_ct_seg.nii.gz')
    
    if comparison == 3: 
        output_path=os.path.join(main_path,"Registrations",experiment_CTseg,patient+'_ct')
    else:
        output_path=os.path.join(main_path,"Registrations",experiment_recon,patient+'_mri')
    try:
        os.makedirs(output_path)
    except OSError:
        print ("Output full folder already exists")
    
    if comparison == 3:
        recon_nib = nib.load(recon_path)
    if comparison == 1:
        gt_nib = nib.load(gt_path)
    
    
    if comparison == 3:
        recon_np = recon_nib.get_fdata().astype("int8")
    if comparison == 1:
        gt_np = gt_nib.get_fdata().astype("int8")
    
    patch_size = 150
    if comparison == 3:
        recon_np = np.pad(recon_np, ((patch_size, patch_size), (patch_size, patch_size),(patch_size, patch_size)), 'constant', constant_values=0)
    if comparison == 1:
        gt_np = np.pad(gt_np, ((patch_size, patch_size), (patch_size, patch_size),(patch_size, patch_size)), 'constant', constant_values=0)
    
    #registered_spine = np.zeros_like(gt_np)
    #cleant_gt = np.zeros_like(gt_np)
    
    vertebrae_dice = []
    
    if comparison==3:
        verts_recon = verts_ct.copy()
    
    error_flag = None
    for idx in range(len(verts_recon)):
        print("Vertebra {}".format(verts_gt[idx]))
        if comparison != 1:
            gt_path = os.path.join(main_path, gt_dataset,patient+'_gt',"Segmentation_L"+str(verts_gt[idx])+'.nii.gz')
            if not os.path.exists(gt_path): 
                print("file not found")
                continue
            gt_nib = nib.load(gt_path)
            gt_np = gt_nib.get_fdata().astype("int8")
            if gt_np.shape[0]>patch_size:
                cut = int((gt_np.shape[0]-patch_size)/2)+1
                gt_np=gt_np[cut:-cut,:,:]
            if gt_np.shape[1]>patch_size:
                cut = int((gt_np.shape[1]-patch_size)/2)+1
                gt_np=gt_np[:,cut:-cut,:]
            if gt_np.shape[2]>patch_size:
                cut = int((gt_np.shape[2]-patch_size)/2)+1
                gt_np=gt_np[:,:,cut:-cut]
            padx_up = int((patch_size-gt_np.shape[0])/2)
            padx_down = int(patch_size-gt_np.shape[0]-padx_up)
            pady_up = int((patch_size-gt_np.shape[1])/2)
            pady_down = int(patch_size-gt_np.shape[1]-pady_up)
            padz_up = int((patch_size-gt_np.shape[2])/2)
            padz_down = int(patch_size-gt_np.shape[2]-padz_up)
            gt_patch = np.pad(gt_np, ((padx_down, padx_up), (pady_down, pady_up),(padz_down, padz_up)), 'constant', constant_values=0)
        
        if comparison == 3:
            recon_patch = recon_np.copy()
            recon_patch[recon_patch!=verts_recon[idx]] = 0
            recon_patch[recon_patch>0] = verts_gt[idx]
            recon_patch, recon_position = extract_patch(recon_patch, patch_size)
        
        else: 
            recon_path = os.path.join(main_path, "MRI_reconstructions",experiment_recon,patient+'_mri',str(verts_recon[idx])+recon_quality+'.nii.gz')
            if not os.path.exists(gt_path):
                print("file not found")
                continue
            recon_nib = nib.load(recon_path)
            recon_patch = recon_nib.get_fdata().astype("int8")
            recon_patch = np.flip(recon_patch,axis=1)
            recon_patch[recon_patch>0] = verts_gt[idx]
            pad = int((patch_size-recon_patch.shape[0])/2)
            recon_patch = np.pad(recon_patch, ((pad, pad), (pad, pad),(pad, pad)), 'constant', constant_values=0)
    
        print("padding") 
        if comparison == 1:
            gt_patch = gt_np.copy()
            gt_patch[gt_patch!=verts_gt[idx]] = 0
            gt_patch, gt_position = extract_patch(gt_patch,patch_size)
        
        recon_patch, dice, error_flag = allign_recon(recon_patch, gt_patch, verts_gt[idx])
        print("allignment..")
        print(dice)
        vertebrae_dice.append(dice)
        
        
        if comparison == 3:
            nib.save(nib.Nifti1Image(recon_patch,affine=None), output_path+'/L'+str(verts_gt[idx])+'_ct.nii.gz')
        else:
            nib.save(nib.Nifti1Image(recon_patch,affine=None), output_path+'/L'+str(verts_gt[idx])+'_mri'+recon_quality+'.nii.gz')
        ''' 
        ME 
        if comparison == 1:
            nib.save(nib.Nifti1Image(gt_patch,affine=None), output_path+'/L'+str(verts_gt[idx])+'_ct.nii.gz')
        else:
            nib.save(nib.Nifti1Image(gt_patch,affine=None), output_path+'/L'+str(verts_gt[idx])+'_gt.nii.gz')
        '''
    
    '''    
        registered_spine[gt_position[0]:gt_position[1],gt_position[2]:gt_position[3],gt_position[4]:gt_position[5]] += recon_patch
        cleant_gt[gt_position[0]:gt_position[1],gt_position[2]:gt_position[3],gt_position[4]:gt_position[5]] += gt_patch
    
    if comparison == 3:
        nib.save(nib.Nifti1Image(registered_spine[patch_size:-patch_size,patch_size:-patch_size,patch_size:-patch_size],affine=None), output_path+'/'+patient+'_ct_seg_registered.nii.gz')
    else:
        nib.save(nib.Nifti1Image(registered_spine[patch_size:-patch_size,patch_size:-patch_size,patch_size:-patch_size],affine=None), output_path+'/'+patient+'_mri_registered.nii.gz')
    if comparison == 1 :
        nib.save(nib.Nifti1Image(cleant_gt[patch_size:-patch_size,patch_size:-patch_size,patch_size:-patch_size],affine=None), output_path+'/'+patient+'_ct_seg.nii.gz')
    else:
        nib.save(nib.Nifti1Image(cleant_gt[patch_size:-patch_size,patch_size:-patch_size,patch_size:-patch_size],affine=None), output_path+'/'+patient+'_ct_gt.nii.gz')
    '''
    
    print("P {} The vertebrae dice scores are: {}".format(patient, vertebrae_dice))
    if error_flag!=None and error_flag:
        print("Error flag was true, at some point the registration could have been better with more shift or rotation")
