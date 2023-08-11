import copy
import json
import os
import shutil
import warnings
from absl import app, flags
import argparse

import torch
#from tensorboardX import SummaryWriter
#from torchvision.datasets import CIFAR10, MNIST
#from torchvision.utils import make_grid, save_image
#from torchvision import transforms
#from tqdm import trange
#from torch.utils.data import DataLoader

from DDPM_Vert.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from DDPM_Vert.model_vert import UNet
#from score.both import get_inception_and_fid_score

#from dataset_vae import CSI_Dataset

modes=['train', 'validation']
"""
FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_string('dataset', 'MNIST', help='dataset name')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 1, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 201, help='total training steps')
flags.DEFINE_integer('img_size', 128, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 1, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_Vert', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_string('train_set',  default='XXX_bodies_data_train',help='name of dataset path')
flags.DEFINE_string('validation_set', default='XXX_bodies_data_validation',help='name of validation-dataset path')
flags.DEFINE_string('test_set', default='XXX_bodies_data_test',help='name of testset path')
flags.DEFINE_bool('test_perf', default=True, help='if prediction should be performed on test dataset')

"""
n_gpus=torch.cuda.device_count()

parser = argparse.ArgumentParser(description='Fully Convolutional Network')
parser.add_argument('--ch', default=128, type=int, help='base channel of UNet')
parser.add_argument('--ch_mult', default=[1, 2, 2, 2], help='channel multiplier')
parser.add_argument('--attn', default=[0], help='add attention to these levels')
parser.add_argument('--num_res_blocks', default=2, help='# resblock in each level')
parser.add_argument('--dropout', default=0.1, help='dropout rate of resblock')

parser.add_argument('--beta_1', default=1e-4, help='start beta value')
parser.add_argument('--beta_T', default=0.02, help='end beta value')
parser.add_argument('--T', default=2, help='total diffusion steps')
parser.add_argument('--mean_type', default='epsilon', choices=['xprev', 'xstart', 'epsilon'], help='predict variable')
parser.add_argument('--var_type', default='fixedlarge', choices=['fixedlarge', 'fixedsmall'], help='variance type')

parser.add_argument('--img_size', default=128, help='image size')
parser.add_argument('--parallel', default=True, help='multi gpu training')

args = parser.parse_args()


for arg in vars(args):
    print(arg, getattr(args, arg))






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of Gpus:", torch.cuda.device_count())

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


class DDPM_model(object):
    def __init__(self):
        # model setup
        self.model = UNet(im_ch=1,
            T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
            num_res_blocks=args.num_res_blocks, dropout=args.dropout)

    def load_model(self, base_model_path):
        # load model and evaluate
        ckpt = torch.load(os.path.join(base_model_path, 'ckpt.pt'))
        self.model.load_state_dict(ckpt['net_model'])
        self.model.eval()

        self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        self.model = self.model.to(device)

        self.sampler = GaussianDiffusionSampler(
            self.model, args.beta_1, args.beta_T, args.T, img_size=args.img_size,
            mean_type=args.mean_type, var_type=args.var_type).to(device)
        #if args.parallel:
        
        self.sampler = torch.nn.parallel.DistributedDataParallel(self.sampler)
        self.sampler = self.sampler.to(device)

    def generation(self, x):
        noise = torch.randn(1, 1, 128, 128, 128)
        noise = noise.to(device)
        batch_images = self.sampler(noise).cpu()
        batch_images = (batch_images + 1) / 2
        
        return batch_images
