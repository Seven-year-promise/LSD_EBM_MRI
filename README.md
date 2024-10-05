## Energy-Based Prior Latent Space Diffusion model for Reconstruction of Lumbar Vertebrae from Thick Slice MRI (LSD-EBM)

### Introduction

This code was origincally created by Aurelio dolfinis, containing the implementation of EBM, LEBM, and our LSD-EBM. The performance of the those models are tested on the public datasets (2D), and our domestic vertebrae datasets (3D). 


### Installation

Create the environment with python 3.7 and install the bpackages required in `requirements.txt`.


### Run the codes

#### 1. Run the codes on 2D datasets

```
cd 2D
```

Run the EBM on 2D datasets and the FID scores as well as geberated samples are generated automatically.

```
cd EBM
python ebm_mnist.py
python ebm_cifar10.py
python ebm_celeba64.py
python ebm_fashion_mnist.py
```

Run the LEBM on 2D datasets and the FID scores as well as geberated samples are generated automatically.

```
cd LEBM
python train_mnist.py
python train_cifar10.py
python train_celeba64.py
python train_fashion_mnist.py
```

Run the LSD-EBM on 2D datasets

```
cd LSD-EBM
python LDEBM_Datasets_NEW_CelebA.py
python LDEBM_Datasets_NEW_CIFAR10.py
python LDEBM_Datasets_NEW_Fashion_MNIST.py
python LDEBM_Datasets_NEW_MNIST.py
```

To generate the FID scores and geberated samples for LSD-EBM, run

```
cd LSD-EBM
python LDEBM_Datasets_NEW_CelebA_Inference.py
python LDEBM_Datasets_NEW_CIFAR10_Inference.py
python LDEBM_Datasets_NEW_FashionMNIST_Inference.py
python LDEBM_Datasets_NEW_MNIST_Inference.py
```

#### 2. Run the codes on 3D datasets

```
cd 3D/LSD-EBM/LSD-EBM_code
```

Run the VAE on 3D datasets

```
main_VAE_Vert_01.py
```

Run the LEBM on 3D datasets

```
python main_LEBM_Vert_04.py
```

Run the LSD-EBM on 3D datasets

```
python main_LSDEBM_Vert_01.py
```

### Analysis


#### 1. Compare the results of different steps of LSD-EBM

```
python compare_steps_lsd_ebm.py
```

#### 2. Compare the latents of LSD-EBM and LEBM

```
python compare_latent_lsdebm_ddpm.py
```

#### 3. Generate the reconstruction metrics used in the paper

```
python reconstruction_metrics.py
```

### Citing our work

```
@inproceedings{wang2024energy,
  title={Energy-Based Prior Latent Space Diffusion model for Reconstruction of Lumbar Vertebrae from Thick Slice MRI},
  author={Wang, Yanke and Lee, Yolanne Y. R. and Dolfini, Aurelio and Reischl, Markus and Konukoglu, Ender and Flouris, Kyriakos},
  booktitle={4th MICCAI Workshop, DGM4MICCAI 2024, Held in Conjunction with MICCAI 2024, Marrakesh, Morocco, October 10, 2024, Proceedings},
  volume={xx},
  pages={xxx},
  year={2024},
  organization={Springer Nature}
}
```
