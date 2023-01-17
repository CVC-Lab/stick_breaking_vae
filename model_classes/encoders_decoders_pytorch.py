import torch
from torch import nn
from utils.util_vars import hidden_ndims, activation, latent_ndims


class GaussianEncoder(object):
    def __init__(self, input_n_dims):
        self.input_to_hidden = nn.Linear(input_ndims, hidden_ndims)
        self.hidden_to_mu = nn.Linear(hidden_ndims, latent_ndims)
        self.hidden_to_sigma = nn.Linear(hidden_ndims, latent_ndims)
        self.activation = activation
        self.encoder_layers = nn.ModuleList([self.input_to_hidden, self.hidden_to_mu, self.hidden_to_sigma])

    def encode(self, x):
        hidden = self.activation(self.input_to_hidden(x))
        parameters = self.hidden_to_mu(hidden), self.hidden_to_sigma(hidden)
        return parameters


class StickBreakingEncoder(object):
    def __init__(self, input_ndims):
        self.input_to_hidden = nn.Linear(input_ndims, hidden_ndims)
        self.hidden_to_alpha = nn.Linear(hidden_ndims, latent_ndims)
        self.hidden_to_beta = nn.Linear(hidden_ndims, latent_ndims)
        self.activation = activation
        self.encoder_layers = nn.ModuleList([self.input_to_hidden, self.hidden_to_alpha, self.hidden_to_beta])
        self.softplus = nn.Softplus()

    def encode(self, x):
        # Softplus per Nalisnick & Smythe github implementation
        hidden = self.activation(self.input_to_hidden(x))
        parameters = self.softplus(self.hidden_to_alpha(hidden)), self.softplus(self.hidden_to_beta(hidden))
        return parameters

class StickBreakingEncoderMSI(object):
    def __init__(self, input_ndims):
        ly = 3
        self.num = 12
        self.nLRlevel = [ly,ly,ly,ly,ly]
        self.nHRlevel = [ly,ly,ly,ly,ly]
        self.msi_input_to_hidden_beta = nn.Linear(input_ndims, self.nLRlevel[0])
        self.msi_beta_hidden_12 = nn.Linear(input_ndims+self.nLRlevel[0], self.nLRlevel[1])
        self.msi_hidden_to_beta = nn.Linear(input_ndims+self.nLRlevel[0]+self.nLRlevel[1], latent_ndims)
        self.alpha = torch.ones(1, latent_ndims) # alpha is fixed to 1
        self.msi_input_to_hidden_uniform = nn.Linear(31, self.nLRlevel[0])
        self.msi_unifom_hidden_12 = nn.Linear(31+self.nLRlevel[0], self.nLRlevel[1])
        self.msi_unifom_hidden_13 = nn.Linear(31+self.nLRlevel[0]+self.nLRlevel[1], self.nLRlevel[2])
        self.msi_unifom_hidden_14 = nn.Linear(31+self.nLRlevel[0]+
                                                self.nLRlevel[1]+self.nLRlevel[2], self.nLRlevel[3])
        self.msi_hidden_to_uniform = nn.Linear(31+self.nLRlevel[0]+
                                                self.nLRlevel[1]+self.nLRlevel[2]+self.nLRlevel[3], self.num)
        self.msi_softplus = nn.Softplus()
        self.msi_encoder_layers = nn.ModuleList([self.msi_input_to_hidden_beta, 
                                                self.msi_beta_hidden_12,
                                                self.msi_hidden_to_beta,
                                                self.msi_input_to_hidden_uniform, 
                                                self.msi_unifom_hidden_12,
                                                self.msi_unifom_hidden_13,
                                                self.msi_unifom_hidden_14,
                                                self.msi_hidden_to_uniform
                                                ])

    def encode_beta_msi(self, x):
        o_11 = self.msi_input_to_hidden_beta(x)
        stack_layer_11 = torch.stack([x, o_11], 1)
        o_12 = self.msi_beta_hidden_12(stack_layer_11)
        stack_layer_12 = torch.stack([stack_layer_11, o_12], 1)
        beta = self.msi_hidden_to_beta(stack_layer_12)
        return beta

    def encode_uniform_msi(self, x):
        o_11 = self.msi_input_to_hidden_uniform(x)
        stack_layer_11 = torch.stack([x, o_11], 1)
        o_12 = self.msi_unifom_hidden_12(stack_layer_11)
        stack_layer_12 = torch.stack([stack_layer_11, o_12], 1)
        o_13 = self.msi_unifom_hidden_13(stack_layer_12)
        stack_layer_13 = torch.stack([stack_layer_12, o_13], 1)
        o_14 = self.msi_unifom_hidden_14(stack_layer_13)
        stack_layer_14 = torch.stack([stack_layer_13, o_14])
        uniform = self.msi_hidden_to_uniform(stack_layer_14)
        return uniform

class StickBreakingEncoderHSI(object):
    def __init__(self, input_ndims, R):
        ly = 3
        self.num = 12
        self.nLRlevel = [ly,ly,ly,ly,ly]
        self.nHRlevel = [ly,ly,ly,ly,ly]
        self.msi_uniform_1 = R
        self.hsi_input_to_hidden_beta = nn.Linear(input_ndims, self.nLRlevel[0])
        self.hsi_beta_hidden_12 = nn.Linear(input_ndims+self.nLRlevel[0], self.nLRlevel[1])
        self.hsi_hidden_to_beta = nn.Linear(input_ndims+self.nLRlevel[0]+self.nLRlevel[1], latent_ndims)
        self.alpha = torch.ones(1, latent_ndims) # alpha is fixed to 1
        self.hsi_input_to_hidden_uniform = nn.Linear(31, self.nLRlevel[0])
        self.hsi_unifom_hidden_12 = nn.Linear(31+self.nLRlevel[0], self.nLRlevel[1])
        self.hsi_unifom_hidden_13 = nn.Linear(31+self.nLRlevel[0]+self.nLRlevel[1], self.nLRlevel[2])
        self.hsi_unifom_hidden_14 = nn.Linear(31+self.nLRlevel[0]+
                                                self.nLRlevel[1]+self.nLRlevel[2], self.nLRlevel[3])
        self.hsi_hidden_to_uniform = nn.Linear(31+self.nLRlevel[0]+
                                                self.nLRlevel[1]+self.nLRlevel[2]+self.nLRlevel[3], self.num)
        self.hsi_softplus = nn.Softplus()
        self.hsi_encoder_layers = nn.ModuleList([self.hsi_input_to_hidden_beta, 
                                                self.hsi_beta_hidden_12,
                                                self.hsi_hidden_to_beta,
                                                self.hsi_input_to_hidden_uniform, 
                                                self.hsi_unifom_hidden_12,
                                                self.hsi_unifom_hidden_13,
                                                self.hsi_unifom_hidden_14,
                                                self.hsi_hidden_to_uniform
                                                ])

    def encode_beta_hsi(self, x):
        o_11 = self.hsi_input_to_hidden_beta(x)
        stack_layer_11 = torch.stack([x, o_11], 1)
        o_12 = self.hsi_beta_hidden_12(stack_layer_11)
        stack_layer_12 = torch.stack([stack_layer_11, o_12], 1)
        beta = self.hsi_hidden_to_beta(stack_layer_12)
        return beta

    def encode_uniform_hsi(self, x):
        x = self.msi_uniform_1.T @ x
        o_11 = self.hsi_input_to_hidden_uniform(x)
        stack_layer_11 = torch.stack([x, o_11], 1)
        o_12 = self.hsi_unifom_hidden_12(stack_layer_11)
        stack_layer_12 = torch.stack([stack_layer_11, o_12], 1)
        o_13 = self.hsi_unifom_hidden_13(stack_layer_12)
        stack_layer_13 = torch.stack([stack_layer_12, o_13], 1)
        o_14 = self.hsi_unifom_hidden_14(stack_layer_13)
        stack_layer_14 = torch.stack([stack_layer_13, o_14])
        uniform = self.hsi_hidden_to_uniform(stack_layer_14)
        return uniform

class Decoder(object):
    def __init__(self, hr_hsi_ndims):
        self.latent_to_hidden = nn.Linear(latent_ndims, 10)
        self.l1 = nn.Linear(10, 10)
        self.hidden_to_reconstruction = nn.Linear(10, hr_hsi_ndims)
        self.dropout = nn.Dropout(p=0.2)
        self.activation = activation
        self.R = nn.Linear(3, 31)
        self.decoder_layers = nn.ModuleList([self.latent_to_hidden, self.l1, self.hidden_to_reconstruction, self.R])
        self.sigmoid = nn.Sigmoid()  # note: used as decoder output activation in the Nalisnick github

    def decode(self, z):
        hidden = self.activation(self.latent_to_hidden(z))
        hidden = self.l1(hidden)
        hidden = self.dropout(hidden)
        reconstruction = self.sigmoid(self.hidden_to_reconstruction(hidden))
        return reconstruction

    def gen_hrhsi(self, recon):
        return self.R(recon.reshape(recon.shape[0], 31, -1).permute(0, 2, 1)).permute(0, 2, 1)
