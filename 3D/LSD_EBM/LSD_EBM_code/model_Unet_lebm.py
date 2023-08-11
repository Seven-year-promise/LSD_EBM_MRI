import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReconNet(nn.Module):
    
    def initial_conv(self, in_channels, out_channels):        
        return nn.Sequential(nn.Conv3d(in_channels, int(out_channels/2), 3, padding = 1),
                nn.BatchNorm3d(int(out_channels/2)),
                nn.ReLU(inplace=True),
                nn.Conv3d(int(out_channels/2), out_channels, 3, padding = 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        
    def consecutive_conv(self, in_channels, out_channels):         
        return nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding = 1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        
    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding = 1), # HEEEERE IT WASS IN OUT
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, padding = 1), #HAND HERE IN IN
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
                
    def __init__(self,num_channels,num_latents):
        super(ReconNet, self).__init__()
        
        print('num_channel',num_channels)
        print('num_latent',num_latents)
        self.num_channels=num_channels      
        self.conv_initial = self.initial_conv(1, num_channels)  
        self.conv_final = nn.Conv3d(num_channels, 1, 3, padding = 1)  
        
        self.conv_rest_x_64 = self.consecutive_conv(num_channels, num_channels*2) 
        self.conv_rest_x_32 = self.consecutive_conv(num_channels*2, num_channels*4) 
        self.conv_rest_x_16 = self.consecutive_conv(num_channels*4, num_channels*8) 

        self.conv_rest_u_32 = self.consecutive_conv_up(num_channels*8+num_channels*4, num_channels*4) 
        self.conv_rest_u_64 = self.consecutive_conv_up(num_channels*4+num_channels*2, num_channels*2) 
        self.conv_rest_u_128 = self.consecutive_conv_up(num_channels*2+num_channels, num_channels) 
        
        #self.linear_enc=nn.Linear(16*16*16*num_channels*8,num_latents)
        #self.linear_dec=nn.Linear(num_latents,16*16*16*num_channels*8)
        
        self.contract = nn.MaxPool3d(2, stride=2) 
        self.expand = nn.Upsample(scale_factor=2)

            
    def forward(self,x):
        x_128 = self.conv_initial(x) #conv_initial 1->16->32
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64) #rest 32->32->64
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32) #rest 64->64->128
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16) #rest 128->128->256

        #x_flat=x_16.view(-1,16*16*16*self.num_channels*8) # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        
        #fc=self.linear_enc(x_flat)
        #u_16 = self.linear_dec(fc).view(-1,self.num_channels*8,16,16,16)

        u_32 = self.expand(x_16)
        u_32 = self.conv_rest_u_32(torch.cat((x_32, u_32),1)) #rest 256+128-> 128 -> 128
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(torch.cat((x_64,u_64),1))  #rest 128+64-> 64 -> 64
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(torch.cat((x_128, u_128),1))  # rest 64+32-> 32 -> 32
        u_128 = self.conv_final(u_128)

        S = torch.sigmoid(u_128)
        
        return S

"""
class VAE_new(nn.Module):
    def initial_conv(self, in_channels, out_channels):        
        return nn.Sequential(nn.Conv3d(in_channels, int(out_channels/2), 3, padding = 1),
                nn.BatchNorm3d(int(out_channels/2)),
                nn.ReLU(inplace=True),
                nn.Conv3d(int(out_channels/2), out_channels, 3, padding = 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        
    def consecutive_conv(self, in_channels, out_channels):         
        return nn.Sequential(nn.Conv3d(in_channels, in_channels, 3, padding = 1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
        
    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding = 1), # HEEEERE IT WASS IN OUT
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, padding = 1), #HAND HERE IN IN
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
                
    def __init__(self,num_channels,num_latents):
        super(VAE_new, self).__init__()
        
        print('num_channel',num_channels)
        print('num_latent',num_latents)
        self.num_channels=num_channels      
        self.conv_initial = self.initial_conv(1, num_channels)  
        self.conv_final = nn.Conv3d(num_channels, 1, 3, padding = 1)  
        
        self.conv_rest_x_64 = self.consecutive_conv(num_channels, num_channels*2) 
        self.conv_rest_x_32 = self.consecutive_conv(num_channels*2, num_channels*4) 
        self.conv_rest_x_16 = self.consecutive_conv(num_channels*4, num_channels*8) 
        
        self.conv_rest_u_32 = self.consecutive_conv_up(num_channels*8, num_channels*4) 
        self.conv_rest_u_64 = self.consecutive_conv_up(num_channels*4, num_channels*2) 
        self.conv_rest_u_128 = self.consecutive_conv_up(num_channels*2, num_channels)
         
        self.contract = nn.MaxPool3d(2, stride=2) 
        self.expand = nn.Upsample(scale_factor=2)
        self.linear_enc=nn.Linear(16*16*16*num_channels*8,num_latents)    
        self.linear_dec=nn.Linear(num_latents,16*16*16*num_channels*8)
        
    def encoder(self, x):
        x_128 = self.conv_initial(x) #conv_initial 1->16->32
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64) #rest 32->32->64
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32) #rest 64->64->128
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16) #rest 128->128->256
        x_flat=x_16.view(-1,16*16*16*self.num_channels*8) # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        mean=self.linear_enc(x_flat)
        std=1.e-6+nn.functional.softplus(self.linear_enc(x_flat))
        return mean,std        
            
    def decoder(self, x):
        u_16=self.linear_dec(x).view(-1,self.num_channels*8,16,16,16)
        u_32 = self.expand(u_16)
        u_32 = self.conv_rest_u_32(u_32) #rest 256+128-> 128 -> 128
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(u_64)  #rest 128+64-> 64 -> 64
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(u_128)  # rest 64+32-> 32 -> 32
        u_128 = self.conv_final(u_128)

        S = torch.sigmoid(u_128)
        return S

    def forward(self,x):
        mean,logvar=self.encoder(x)
        std = torch.exp(0.5*logvar) # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std
        z=mean+std* torch.randn_like(std) # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent 
        return mean,logvar,self.decoder(z) # note that is the logvar that gets returned
"""




"""  
class VAE(nn.Module):
    def consecutive_conv(self, in_channels, out_channels):
        
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels),
                nn.Conv3d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm3d(out_channels))
                
    def __init__(self,num_channels,num_latents):
        super(VAE, self).__init__()
        
        print('num_channel',num_channels)
        print('num_latent',num_latents)
        self.num_channels=num_channels      
        self.conv_initial = self.consecutive_conv(1, num_channels)  
        self.conv_final = self.consecutive_conv(num_channels, 1)    
        self.conv_rest_x_128 = self.consecutive_conv(num_channels, num_channels) 
        self.conv_rest_x_64 = self.consecutive_conv(num_channels, num_channels) 
        self.conv_rest_x_32 = self.consecutive_conv(num_channels, num_channels) 
        self.conv_rest_x_16 = self.consecutive_conv(num_channels, num_channels) 

        self.conv_rest_u_16 = self.consecutive_conv(num_channels, num_channels) 
        self.conv_rest_u_32 = self.consecutive_conv(num_channels, num_channels) 
        self.conv_rest_u_64 = self.consecutive_conv(num_channels, num_channels) 
        self.conv_rest_u_128 = self.consecutive_conv(num_channels, num_channels) 
         
        self.contract = nn.MaxPool3d(2, stride=2) 
        self.expand = nn.Upsample(scale_factor=2) 
        self.linear_enc=nn.Linear(16*16*16*num_channels,num_latents)    
        self.linear_dec=nn.Linear(num_latents,16*16*16*num_channels)
        
    def encoder(self, x):
        x_128 = self.conv_rest_x_128(self.conv_initial(x))
        x_64 = self.conv_rest_x_64(self.contract(x_128))
        x_32 = self.conv_rest_x_32(self.contract(x_64))
        x_16 = self.conv_rest_x_16(self.contract(x_32))
        x_flat=x_16.view(-1,16*16*16*self.num_channels) # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        mean=self.linear_enc(x_flat)
        std=1.e-6+nn.functional.softplus(self.linear_enc(x_flat))
        return mean,std        
            
    def decoder(self, x):
        u_16=self.conv_rest_u_16(self.linear_dec(x).view(-1,self.num_channels,16,16,16))
        u_32 = self.conv_rest_u_32(self.expand(u_16))  
        u_64 = self.conv_rest_u_64(self.expand(u_32))  
        u_128 = self.conv_rest_u_128(self.expand(u_64))  
        u_128 = self.conv_final(u_128)

        S = torch.sigmoid(u_128)
        return S

    def forward(self,x):
        mean,logvar=self.encoder(x)
        std = torch.exp(0.5*logvar) # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std
        z=mean+std* torch.randn_like(std) # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent 
        return mean,logvar,self.decoder(z) # note that is the logvar that gets returned
"""

# PRIOR MODEL
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class E_func(nn.Module):
    def __init__(self, latent_dim=100, inner_dim=200, num_layers=1, out_dim=1, batchnorm=False):
        super(E_func, self).__init__()
        # prior layers
        self.out_dim = out_dim
        self.lat_dim = latent_dim

        apply_sn = lambda x: x
        f = GELU()

        inner_layers = []
        if batchnorm:
            inner_layers += [nn.BatchNorm1d(inner_dim)]
            for cnt in range(num_layers): inner_layers += [apply_sn(nn.Linear(inner_dim, inner_dim)), f, nn.BatchNorm1d(inner_dim)]
        else:
            for cnt in range(num_layers): inner_layers += [apply_sn(nn.Linear(inner_dim, inner_dim)), f]

        layers = [apply_sn(nn.Linear(latent_dim, inner_dim)), f] + inner_layers + [apply_sn(nn.Linear(inner_dim, out_dim))]

        # TODO: here changed
        ebm = []
        for l in layers:
            ebm.append(l)
        self.ebm = nn.Sequential(*ebm)

        """
        self.ebm = nn.Sequential()
        for l in layers:
          self.ebm.append(l)
        """


    def forward(self, z):
        #z = z.squeeze()
        z = z.view(-1, self.lat_dim)
        z = self.ebm(z)
        return  z.view(-1, self.out_dim, 1, 1)

class Buff(nn.Module):
    def __init__(self, elems, size):
        super(Buff, self).__init__()
        self.size = size
        self.buff = elems

    def get_samp(self, n):
        perm = torch.randperm(self.buff.size()[0])
        idx = perm[:n]
        return self.buff[idx]

    def add_samp(self, samples):
        new_buff = torch.cat([samples, self.buff])
        self.buff = new_buff[:self.size]

class GenModel_vert(nn.Module):
    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, padding = 1), # HEEEERE IT WASS IN OUT
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, 3, padding = 1), #HAND HERE IN IN
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True))
                
    def __init__(self,num_channels,num_latents):
        super(GenModel_vert, self).__init__()
        
        print('num_channel',num_channels)
        print('num_latent',num_latents)
        self.num_channels=num_channels      
        self.conv_final = nn.Conv3d(num_channels, 1, 3, padding = 1)  
        
        self.conv_rest_u_32 = self.consecutive_conv_up(num_channels*8, num_channels*4) 
        self.conv_rest_u_64 = self.consecutive_conv_up(num_channels*4, num_channels*2) 
        self.conv_rest_u_128 = self.consecutive_conv_up(num_channels*2, num_channels)
         
        self.expand = nn.Upsample(scale_factor=2)
        self.linear_dec=nn.Linear(num_latents,16*16*16*num_channels*8)  
            
    def forward(self, x):
        u_16=self.linear_dec(x).view(-1,self.num_channels*8,16,16,16)
        u_32 = self.expand(u_16)
        u_32 = self.conv_rest_u_32(u_32) #rest 256+128-> 128 -> 128
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(u_64)  #rest 128+64-> 64 -> 64
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(u_128)  # rest 64+32-> 32 -> 32
        u_128 = self.conv_final(u_128)

        S = torch.sigmoid(u_128)
        return S


# Sampling
mse = nn.MSELoss(reduction='sum')

def sample_langevin_prior_z(z0, netE, e_l_steps=60, e_l_step_size = 0.4, e_prior_sig = 1., noise_sig=1., gamma=1., e_l_with_noise = True, verbose=False):
        # z -> m x latent_dim
        netE.eval()
        for p in netE.parameters():
            p.requires_grad = False
        z = z0.clone().detach()
        z.requires_grad = True
                
        noise = torch.randn(z.shape, device=z.device)

        for i in range(e_l_steps):                        # mcmc iteration 
            
            # calculate gradients for the current input.
            en_z = netE.forward(z)
            en_z.sum().backward()

            # Additional to the gradE there is the term steming from p0(z) in p_alpha(z)!
            z.data.add_(- 0.5 * e_l_step_size * (z.grad.data + 1.0 / (e_prior_sig * e_prior_sig) * z.data))

            if e_l_with_noise:
                noise.normal_(0, noise_sig)
                z.data.add_(np.sqrt(e_l_step_size) * noise.data)

            z.grad.detach_()
            z.grad.zero_()

            e_l_step_size *= gamma

        for p in netE.parameters():
            p.requires_grad = True
        netE.train()
        
        return z.detach() 
                     
def sample_langevin_post_z(z0, x, netG, netE, g_l_steps=20, g_l_step_size=0.1, e_prior_sig=1.,  noise_sig=1., g_llhd_sigma=0.3, gamma=1., g_l_with_noise=True, verbose=False):
    netE.eval()
    netG.eval()
    for p in netE.parameters():
        p.requires_grad = False
    for p in netG.parameters():
        p.requires_grad = False
    z = z0.clone().detach()
    z.requires_grad = True
    
    noise = torch.randn(z.shape, device=z.device)
    
    for i in range(g_l_steps):        # mcmc iteration

        x_hat = netG.forward(z)               # x = gen_nw(z)
        # log lkhd (x|z) = 1/2 * (x-x_true) / sigma^2
        log_lkhd = 1.0 / (2.0 * g_llhd_sigma * g_llhd_sigma) * mse(x_hat, x)
        grad_log_lkhd = torch.autograd.grad(log_lkhd, z)[0]
        
        en = netE.forward(z)                     # E(z)
        grad_log_prior_pt1 = torch.autograd.grad(en.mean(), z)[0]
        grad_log_prior_pt2 = 1.0 / (e_prior_sig * e_prior_sig) * z.data.detach()

        # gradLogP(z|x) = gradLogP(x|z) + gradLogP(z) - gradLogP(x) with last term =0
        z.data = z.data - 0.5 * g_l_step_size * (grad_log_lkhd + grad_log_prior_pt1 + grad_log_prior_pt2)
        # z.data.add_(- 0.5 * g_l_step_size * (z_grad_g + z_grad_e + 1.0 / (e_prior_sig * e_prior_sig) * z.data))
        # .. + s * N(0,1).sample()
        if g_l_with_noise:
            noise.normal_(0, noise_sig)
            z.data.add_(np.sqrt(g_l_step_size) * torch.randn_like(z).data)

        g_l_step_size *= gamma

    for p in netE.parameters():
        p.requires_grad = True
    for p in netG.parameters():
        p.requires_grad = True
    netE.train()
    netG.train()

    return z.detach() 


