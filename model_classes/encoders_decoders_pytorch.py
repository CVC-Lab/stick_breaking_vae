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
        self.msi_input_to_hidden = nn.Linear(input_ndims, 5)
        self.msi_l1 = nn.Linear(5, 7)
        self.msi_l2 = nn.Linear(7, 9)
        self.msi_hidden_to_beta = nn.Linear(9, latent_ndims)
        self.alpha = torch.ones(1, latent_ndims) # alpha is fixed to 1
        self.msi_activation = activation
        self.msi_encoder_layers = nn.ModuleList([self.msi_input_to_hidden, 
                                                self.msi_l1,
                                                self.msi_l2, 
                                                self.msi_hidden_to_beta])
        self.msi_softplus = nn.Softplus()

    def encode_msi(self, x):
        # Softplus per Nalisnick & Smythe github implementation
        hidden = self.msi_activation(self.msi_input_to_hidden(x))
        hidden = self.msi_activation(self.msi_l1(hidden))
        hidden = self.msi_activation(self.msi_l2(hidden))
        parameters = self.alpha, self.msi_softplus(self.msi_hidden_to_beta(hidden))
        return parameters

class StickBreakingEncoderHSI(object):
    def __init__(self, input_ndims):
        self.hsi_input_to_hidden = nn.Linear(input_ndims, 10)
        self.hsi_l1 = nn.Linear(10, 10)
        self.hsi_hidden_to_beta = nn.Linear(10, latent_ndims)
        self.alpha = torch.ones(1, latent_ndims).cuda() # alpha is fixed to 1
        self.hsi_activation = activation
        self.hsi_encoder_layers = nn.ModuleList([self.hsi_input_to_hidden, 
                                                self.hsi_l1,
                                                self.hsi_hidden_to_beta])
        self.hsi_softplus = nn.Softplus()

    def encode_hsi(self, x):
        # Softplus per Nalisnick & Smythe github implementation
        hidden = self.hsi_activation(self.hsi_input_to_hidden(x))
        hidden = self.hsi_activation(self.hsi_l1(hidden))
        parameters = self.alpha, self.hsi_softplus(self.hsi_hidden_to_beta(hidden))
        return parameters

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

    def gen_hrmsi(self, recon):
        return self.R(recon.reshape(recon.shape[0], 31, -1).permute(0, 2, 1)).permute(0, 2, 1)
