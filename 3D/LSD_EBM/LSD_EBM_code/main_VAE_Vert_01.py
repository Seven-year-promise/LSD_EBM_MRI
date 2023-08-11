from __future__ import print_function, division
import os
import argparse
import time
import pickle
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.optim as optim
from model_Unet_vae import ReconNet, VAE_new
from dataset_vae import CSI_Dataset
import nibabel as nib
import torch.nn as nn
from medpy.metric.binary import dc


modes=['train','validation']

def seg_loss(pred, target, weight, weightFP):
    
    FP = torch.sum(weightFP*(1-target)*pred)
    FN = torch.sum(weight*(1-pred)*target)

    return FP, FN

def VaeLoss(recon_x,x,mu,logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')#/10000000
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())# * 100.
    print("BCE: %.4f\tKLD: %.4f\t" % (BCE.item()/batch_size, KLD.item()/batch_size))
    loss = BCE+KLD
    return loss

def DiceCoeff(pred, gt):
     pred = pred.to('cpu').numpy()
     gt = gt.to('cpu').numpy()
     
     #if gt is all zero (use inverse to count)
     if np.count_nonzero(gt) == 0:
         gt = gt+1
         pred = 1-pred
         
     dice_score = dc(pred.astype(np.int),gt.astype(np.int)) 
     return dice_score
 
def DiceCoeffProc(pred, gt, body):
    pred = pred.to('cpu').numpy()
    gt = gt.to('cpu').numpy()
    body = body.to('cpu').numpy()
    
    pred = pred-body
    gt = gt-body
 
    #if gt is all zero (use inverse to count)
    if np.count_nonzero(gt) == 0:
        gt = gt+1
        pred = 1-pred
     
    dice_score = dc(pred.astype(np.int),gt.astype(np.int)) 
    return dice_score


def load_fixed(img_names, dataset_path):
    full=[]
    body=[]
    for img_name in img_names:
        body_name = img_name.split('.')[0]+'_weight.nii.gz'

        img_file = os.path.join(dataset_path,  'full',  img_name)
        full_patch = np.round(nib.load(img_file).get_fdata()).astype('uint8')
        
        body_file = os.path.join(dataset_path, 'border',  body_name)
        body_patch = np.round(nib.load(body_file).get_fdata()).astype('uint8')
        
        full_patch = np.expand_dims(full_patch.astype('uint8'), axis=(0,1))
        body_patch = np.expand_dims(body_patch.astype('uint8'), axis=(0,1))
        
        full.append(torch.from_numpy(full_patch))
        body.append(torch.from_numpy(body_patch))
    
    return full, body




#%%Main
if  __name__ == "__main__" :   
    
    fixed_images = ['L1_verse033_seg.nii.gz', 'L1_verse265_seg.nii.gz', 'L2_verse145_seg.nii.gz',  'L3_verse082_seg.nii.gz']
    
    # Training args
    parser = argparse.ArgumentParser(description='Fully Convolutional Network')
    
    parser.add_argument('--num_epochs',                         default=201, help='Number of epochs')                      
    parser.add_argument('--vae_lr', type=float,                 default=0.00002,help='learning rate (default: 0.001)')
    parser.add_argument('--save_model', action='store_true',    default=True,help='For Saving the current Model')
    parser.add_argument('--train_set',                          default='XXX_bodies_data_train',help='name of dataset path')
    parser.add_argument('--validation_set',                     default='XXX_bodies_data_validation',help='name of validation-set path')
    parser.add_argument('--test_set',                           default='XXX_bodies_data_test',help='name of testset path')
    parser.add_argument('--experiment',                         default='Test',help='name of experiment')
    parser.add_argument('--load_from_ep',                       default=None, type=int, help='checkpoint you want to load for the models')
    parser.add_argument('--epoch_start',                        default=0, type=int, help='epoch you want to start from')
    parser.add_argument('--save_visu',                          default=True, help='saves on training/testing visualization')
    parser.add_argument('--num_channels',                       default=16, type=int,help='Number of Channels for the CNN')
    parser.add_argument('--num_latents',                        default=100, type=int,help='dimension of latent space')
    parser.add_argument('--batch_size',                         default=4, type=int,help='Batch Size')
    # Testing
    parser.add_argument('--test_perf',                          default=True, help='if prediction should be performed on test dataset')
    args = parser.parse_args()

    run_name = "VAE_Vert_01"
    main_path ="./../Vert_data/" + run_name + "/" 

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    with open(main_path + "hyper_para.txt", "w") as output:  ## creates new file but empty
        for arg in vars(args):
            print(arg, getattr(args, arg))
            output.write(str(arg) + "\t" + str(getattr(args, arg)) + "\n")
    
    out_f = open(main_path + "output.txt",'w')
    
    print("Pytorch Version:", torch.__version__, file=out_f)
    print("Experiment: "+args.experiment, file=out_f)
    print(args, file=out_f)
    
    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    
    # Initialize networks
    vaeNet=VAE_new(args.num_channels,args.num_latents).to(device) # Since the images are b&w, they initially have 1 ch
        
    # use multiple gpus
    n_gpus=torch.cuda.device_count()
    if n_gpus>1:
        print("Let's use", n_gpus, "GPUs!", file=out_f)
        batch_size = n_gpus*args.batch_size
        disNet=nn.DataParallel(VAE_new(args.num_channels,args.num_latents).to(device))
        
    #load models if requested
    if args.load_from_ep is not None:
        vaeNet.load_state_dict(torch.load(os.path.join(main_path,'experiments', 'vae_'+str(args.load_from_ep)+'.pth')))
        print("LOAD VAE MODEL EP ", args.load_from_ep, file=out_f)

    print(vaeNet)
    p = 0
    for pms in vaeNet.parameters():
        p += torch.numel(pms)
    print("vaeNet num parameters: ", p, file=out_f)

    # optimizesr
    optimizerVae = optim.Adam(vaeNet.parameters(), lr=args.vae_lr)
    
    
    train_root = os.path.join(main_path,"..", args.train_set)
    validation_root = os.path.join(main_path,"..", args.validation_set)
    test_root = os.path.join(main_path, "..", args.test_set)
    visu_path=os.path.join(main_path,'visu',args.experiment)
    
    
    fixed_full, fixed_body = load_fixed(fixed_images, test_root)
    for idx in range(len(fixed_full)):
        fixed_full[idx] = fixed_full[idx].to(device=device, dtype=torch.float)
        fixed_body[idx] = fixed_body[idx].to(device=device, dtype=torch.float)
        
        
    imgs_list_train     =os.listdir(os.path.join(train_root,'full'))
    imgs_list_validation=os.listdir(os.path.join(validation_root,'full'))
    imgs_list_test      =os.listdir(os.path.join(test_root,'full'))
    
    list_length_train       =len(imgs_list_train)
    list_length_validation  =len(imgs_list_validation)
    list_length_test        =len(imgs_list_test)
    print("Training set size: "    , list_length_train, file=out_f)
    print("Validation set size: "  , list_length_validation, file=out_f)
    print("Test set size: "        , list_length_test, file=out_f)
    
    #create variables as dictionaries
    dataset={}
    loader={}
    vae_loss_dic={}
    v_dice_dic={}
    v_dice_proc_dic={}
    
    
    for i, mode in enumerate(modes+["test"]):
        if mode == 'train':
            dataset[mode]=CSI_Dataset(train_root, mode=mode)
            if batch_size>list_length_train:
                loader[mode]=DataLoader(dataset[mode], batch_size=list_length_train, shuffle=True, drop_last=True)
            else:
                loader[mode]=DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=True)
        elif mode == 'validation':
            dataset[mode]=CSI_Dataset(validation_root, mode=mode)
            if batch_size>list_length_validation:
                loader[mode]=DataLoader(dataset[mode], batch_size=list_length_validation, shuffle=True, drop_last=True)
            else:
                loader[mode]=DataLoader(dataset[mode], batch_size=batch_size, shuffle=True, drop_last=True)
        elif mode == 'test' and args.test_perf == True:
            dataset[mode]=CSI_Dataset(test_root, mode=mode)
            loader[mode]=DataLoader(dataset[mode], batch_size=1, shuffle=True, drop_last=True)
        
        vae_loss_dic[mode]=[]
        v_dice_dic[mode]=[]
        v_dice_proc_dic[mode]=[]
                        
    
    # Start Training/testing
    for epoch in range(args.epoch_start,args.epoch_start+args.num_epochs):

        start_time = time.time()
        
        epoch_v_loss = {}
        epoch_v_dice = {}
        epoch_v_dice_proc = {}
        
                
        for mode in modes:
            
            if mode=='train':
                vaeNet.train()
            else:
                vaeNet.eval()

            
            epoch_v_loss[mode]=[]
            epoch_v_dice[mode]=[]
            epoch_v_dice_proc[mode]=[]


            for i, data in enumerate(loader[mode], 0):
                torch.cuda.empty_cache()
                
                img_name, full_patch, body_patch = data
                
                full_patch = full_patch.to(device=device, dtype=torch.float)
                '''
                if i == 5:
                    for g in optimizerVae.param_groups:
                        g['lr'] = 0.0001
                ''' 
                ############################
                # (1) Update VAE network
                ############################
                    
                if mode=='train':
                    optimizerVae.zero_grad()
                
                mean,logvar,vae_recon_patch = vaeNet(full_patch)
                
                #mean,logvar=vaeNet.encoder(vae_input_patch)
                #vae_recon_patch = vaeNet.decoder(mean)
                
                vae_recon_patch = vae_recon_patch.to(device=device, dtype=torch.float)
                 
                vae_loss = VaeLoss(vae_recon_patch,full_patch, mean,logvar)/batch_size
            
                vae_loss.backward()
                
                if mode=='train':
                    optimizerVae.step()
                    
                    
                with torch.no_grad(): 
                    vae_pred_patch = torch.round(vae_recon_patch).detach() 
                    vae_dice = DiceCoeff(vae_pred_patch, full_patch.detach()) #Note that the recon itself is not binary
                    #vae_dice_proc = DiceCoeffProc(vae_pred_patch, full_patch.detach(), body_patch.detach()) #Note that the recon itself is not binary
                    vae_dice_proc = 0.

                print('[%d/%d][%d/%d]\tVae_l: %.4f\tV_Dice: %.4f\tV_DiceProc: %.4f\n'% (epoch, args.num_epochs, i, len(loader[mode]),vae_loss.item(), vae_dice, vae_dice_proc), file=out_f)
                
                epoch_v_loss[mode].append(vae_loss.item())
                epoch_v_dice[mode].append(vae_dice)
                epoch_v_dice_proc[mode].append(vae_dice_proc)
                
           
                if args.save_visu and (epoch%20==0):
                    patch_path=os.path.join(visu_path,str(epoch),mode,'patch_'+str(i)+'_'+img_name[0][:-7])
                    try:
                        os.makedirs(patch_path)
                    except OSError:
                        None
                    with torch.no_grad():
                        nib.save(nib.Nifti1Image(np.squeeze(vae_pred_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/vae_recon_patch.nii.gz')
                        nib.save(nib.Nifti1Image(np.squeeze(full_patch.to('cpu').numpy()[0]),affine=None), patch_path+ '/full_patch.nii.gz')
                        
                    with open(os.path.join(visu_path,str(epoch),mode,"loss.txt"), "a") as f:
                        f.write("\nPatch {}:    Vaeloss: {:.6}    VaeDice: {:.6}".format(str(i), vae_loss.item(), vae_dice))
                    with open(os.path.join(visu_path,str(epoch),mode,"loss.csv"), "a") as f:
                        f.write("{};{};{}".format(str(i), vae_loss.item(), vae_dice))

                
            vae_loss_dic[mode].append(epoch_v_loss[mode])
            v_dice_dic[mode].append(epoch_v_dice[mode])
            v_dice_proc_dic[mode].append(epoch_v_dice_proc[mode])
            
            
            avg_v_loss = sum(epoch_v_loss[mode]) / len(epoch_v_loss[mode])
            avg_v_dice = sum(epoch_v_dice[mode]) / len(epoch_v_dice[mode])
            avg_v_dice_proc = sum(epoch_v_dice_proc[mode]) / len(epoch_v_dice_proc[mode])
            
            print('\n{} Epoch: {} \t  VaeLoss: {:.4f} \t VaeDice: {:.4} \t VaeDiceProc: {:.4}%\n\n'.format(mode,epoch, avg_v_loss, avg_v_dice, avg_v_dice_proc), file=out_f)
            
            #if args.save_visu and (epoch%50==0):
            with open(os.path.join(visu_path,"epoch_loss.txt"), "a") as f:
                f.write("Epoch {}: {}    Vloss: {}    VaeDice:{}    VaeDiceProc:{} \n".format(epoch,mode,avg_v_loss,avg_v_dice, avg_v_dice_proc))
                if mode == 'validation':
                    print("\n")
            with open(os.path.join(visu_path,"epoch_loss.csv"), "a") as f:
                f.write("{};{};{};{};{} \n".format(epoch, mode, avg_v_loss, avg_v_dice, avg_v_dice_proc))
       
        if args.save_visu and (epoch%20==0):
            with torch.no_grad():
                for idx in range(len(fixed_full)):
                    print("Fixed image: " + fixed_images[idx], file=out_f)
                    patch_path=os.path.join(visu_path,str(epoch),'fixed',fixed_images[idx][:-7])
                    try:
                        os.makedirs(patch_path)
                    except OSError:
                        None
                    mean,logvar,fixed_vae_recon = vaeNet(fixed_full[idx])
                    fixed_vae_recon = fixed_vae_recon.to(device=device, dtype=torch.float)
                    fixed_vae_loss = VaeLoss(fixed_vae_recon, fixed_full[idx], mean,logvar)/batch_size
                    fixed_vae_pred = torch.round(fixed_vae_recon).detach()
                    fixed_vae_dice = DiceCoeff(fixed_vae_pred, fixed_full[idx].detach())
                    fixed_vae_dice_proc = DiceCoeffProc(fixed_vae_pred,  fixed_full[idx].detach(),  fixed_body[idx].detach()) #Note that the recon itself is not binary
                    
                    nib.save(nib.Nifti1Image(np.squeeze(fixed_vae_pred.to('cpu').detach().numpy()),affine=None),patch_path+ '/vae_recon_patch.nii.gz')
                    nib.save(nib.Nifti1Image(np.squeeze(fixed_full[idx].to('cpu').detach().numpy()),affine=None), patch_path+ '/full_patch.nii.gz')   
                    with open(os.path.join(visu_path,str(epoch),'fixed',"recon_loss.txt"), "a") as f:
                        f.write(fixed_images[idx] + "\tVloss:{}\tDice:{}\tDiceProc:{} \n".format(fixed_vae_loss,fixed_vae_dice, fixed_vae_dice_proc))
                    with open(os.path.join(visu_path,str(epoch),'fixed',"recon_loss.csv"), "a") as f:
                        f.write("{};{};{};{};{} \n".format(epoch, mode, fixed_vae_loss, fixed_vae_dice, fixed_vae_dice_proc))
 
        #Saving current model
        if epoch%20==0:
            model_path = os.path.join(main_path,'experiments')
            if not os.path.exists(model_path):
                print('Creating directory at:', model_path, file=out_f)
                os.makedirs(model_path)

            torch.save(vaeNet.state_dict(),os.path.join(main_path,'experiments','vae_'+str(epoch)+'.pth'))
            with open(os.path.join(main_path,"experiments",args.experiment+".p"), 'wb') as f:
                pickle.dump([vae_loss_dic,v_dice_dic,v_dice_proc_dic],f)
            print("--- %s seconds ---" % (time.time() - start_time), file=out_f)


    # TEST PERFORMANCE
    if args.test_perf:
        
        # (A) EVALUATION ON TEST-SPLIT OF CURRENT DATASET
        vaeNet.eval()
        dice_scores = []

        for i, data in enumerate(loader["test"], 0):
            torch.cuda.empty_cache()
            img_name, full_patch, body_patch = data
            full_patch = full_patch.to(device=device, dtype=torch.float)
                
            mean,logvar,vae_recon_patch = vaeNet(full_patch)
            pred_patch = torch.round(vae_recon_patch).detach() 
            
            dice = DiceCoeff(pred_patch, full_patch.detach())
            print(i,"/",len(loader["test"]),": Dice=", dice, file=out_f) 
            dice_scores.append(dice)
            
            # store recon
            patch_path = os.path.join(visu_path,"test_set",'patch_'+img_name[0][:-7])
            try:
                os.makedirs(patch_path)
            except OSError:
                None
            nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/recon_patch.nii.gz')
            nib.save(nib.Nifti1Image(np.squeeze(full_patch.to('cpu').numpy()[0]),affine=None),patch_path+ '/full_patch.nii.gz')
            
            del vae_recon_patch
            del pred_patch
            del full_patch
        print(dice_scores, file=out_f)
        print(np.array(dice_scores).mean(), file=out_f)
        print(np.array(dice_scores).std(), file=out_f)
        """ 
        # (B) EVALUATION ON ADDITIONAL DATASET
        print("### Make predictions on test dataset")
        run_name = "VAE_Vert_01_r"
        experiment_recon = 'recon4_augmheavy'
        experiment_recon_save = 'my_recon4_augmheavy'
        patients = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']
        verts_recon = [1,2,3,4,5]
        testds_path = "./MRI_reconstructions/" 
        save_path = "/scratch_net/biwidl212/adolfini/vert_data/eval_test/MRI_reconstructions/"
        
        for patient in patients:
            print("## Patient {}".format(patient))
            for idx in range(len(verts_recon)):
                print("# Vertebra {}".format(verts_recon[idx]))
            
                hq_path  = os.path.join(testds_path, experiment_recon,patient+'_mri',str(verts_recon[idx])+'_HQ.nii.gz')
                out_path = os.path.join(save_path, experiment_recon_save,patient+'_mri',str(verts_recon[idx])+'_'+run_name +'.nii.gz')
                if not os.path.exists(hq_path):
                    print("file not found")
                    continue

                hq_nib = nib.load(hq_path)
                hq_patch = hq_nib.get_fdata().astype("int8")
                print(hq_patch.max())
                hq_patch[hq_patch!=0]=1
                print(hq_patch.min())
                print(hq_patch.max())
                hq_patch = torch.from_numpy(hq_patch).view(1,1,128,128,128).to(device=device, dtype=torch.float)

                # reconstruction
                mean,logvar,vae_recon_patch = vaeNet(hq_patch)
                pred_patch = torch.round(vae_recon_patch).detach() 
                nib.save(nib.Nifti1Image(np.squeeze(pred_patch.to('cpu').numpy()[0]),affine=None), out_path)    
        """        
