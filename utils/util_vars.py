import torch
import os
import torchvision
import numpy as np
from torch import nn
from torchvision import transforms
import pdb

# arg defaults in https://github.com/enalisnick/stick-breaking_dgms/blob/master/run_gauss_VAE_experiments.py
seed = 1234
torch.set_rng_state(torch.manual_seed(seed).get_state())
batch_size = 2
latent_ndims = 50
hidden_ndims = 500
learning_rate = 1e-4
lookahead = 30
print_interval = 10
n_train_epochs = 1500
init_weight_mean_var = (0, .001)
prior_mu = torch.Tensor([0.])
glogit_prior_mu = torch.Tensor([-1.6])
prior_sigma = torch.Tensor([1.])
prior_alpha = torch.Tensor([1.])
prior_beta = concentration_alpha0 = torch.Tensor([5.])
uniform_low = torch.Tensor([.01])
uniform_high = torch.Tensor([.99])
activation = nn.ReLU()
train_valid_test_splits = (45000, 5000, 10000)
n_monte_carlo_samples = 1
n_random_samples = 16
dataloader_kwargs = {}
download_needed = not os.path.exists('./MNIST')
model_path = 'trained_models'
checkpoint_path = None  # specify path to continue training a model

# use GPU, if available
CUDA = torch.cuda.is_available()
if CUDA:
    torch.cuda.manual_seed(seed)
    dataloader_kwargs.update({'num_workers': 1, 'pin_memory': True})

# get datasets
import sys
sys.path.append("../../../")
from hmi_fusion.datasets.cave_dataset import CAVEDataset, R
train_dataset = CAVEDataset("/data/shubham/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/CAVE", cl="balloons_ms", mode="train")
test_dataset = CAVEDataset("/data/shubham/HSI-MSI-Image-Fusion/hmi_fusion/datasets/data/CAVE", cl="balloons_ms", mode="test")
# from ....hmi_fusion.datasets.harvard_dataset import HarvardDataset
lr_hsi_shape, hr_msi_shape, hr_hsi_shape = train_dataset[0][0].shape, train_dataset[0][1].shape, train_dataset[0][2].shape # Yh, Ym, X
lr_hsi_ndims, hr_msi_ndims = np.product(lr_hsi_shape), np.product(hr_msi_shape)
hr_hsi_ndims = np.product(hr_hsi_shape)
# pdb.set_trace()
# msi_dims, hsi_dims = train_dataset[0]
# get dimension info
# input_shape = list(train_dataset.data[0][0].shape)
# input_ndims = np.product(input_shape)

# # define data loaders
# train_data = train_dataset.data.reshape(-1, 1, *input_shape) / 255  # reshaping and scaling bytes to [0,1]
# test_data = test_dataset.data.reshape(-1, 1, *input_shape) / 255
# pruned_train_data = train_data[:train_valid_test_splits[0]]
# train_loader = torch.utils.data.DataLoader(pruned_train_data,
#                                            shuffle=True, batch_size=batch_size, **dataloader_kwargs)
# test_loader = torch.utils.data.DataLoader(test_data,
#                                           shuffle=False, batch_size=batch_size, **dataloader_kwargs)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True, batch_size=batch_size, **dataloader_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=False, batch_size=batch_size, **dataloader_kwargs)

parametrizations = dict(Kumar='Kumaraswamy', GLogit='Gauss_Logit', GEM='GEM')