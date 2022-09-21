import torch
# import botorch
from model_classes.encoders_decoders_pytorch import GaussianEncoder, StickBreakingEncoder, Decoder
from utils.util_vars import init_weight_mean_var, latent_ndims, prior_mu, prior_sigma, prior_alpha, \
    prior_beta, uniform_low, uniform_high, glogit_prior_mu
from utils.util_funcs import beta_func, logistic_func
from numpy.testing import assert_almost_equal
import pdb

class VAE(object):
    def __init__(self, target_distribution, latent_distribution, prior_params):
        self.target_distribution = target_distribution
        self.latent_distribution = latent_distribution
        # self.prior_param1 = prior_param1
        # self.prior_param2 = prior_param2
        self.prior_params = prior_params

        # self.init_weights(self.encoder_layers)
        # self.init_weights(self.decoder_layers)

        self.uniform_distribution = torch.distributions.uniform.Uniform(low=uniform_low, high=uniform_high)

    def init_weights(self, layers):
        for layer in layers:
            with torch.no_grad():
                layer_size = (layer.out_features, layer.in_features)
                layer.weight = torch.nn.Parameter(torch.normal(*init_weight_mean_var, layer_size))
                layer.bias = torch.nn.Parameter(torch.zeros(layer.out_features))

    def ELBO_loss(self, recon_x, x, param1, param2, kl_divergence):
        n_samples = len(recon_x)
        x = x.view(-1, input_ndims)
        if not torch.isfinite(recon_x.log()).all():
            raise AssertionError('Reconstructed x.log not finite!: ', recon_x.log())
        reconstruction_loss = - (x * recon_x.log() + (1 - x)
                                 * (1 - recon_x).log()).sum(axis=1)  # per Nalisnick & Smyth github
        regularization_loss = torch.stack([kl_divergence(param1[i], param2[i]) for i in range(n_samples)])
        return reconstruction_loss.mean(),  regularization_loss.mean()

    def reparametrize(self, param1, param2, parametrization=None):
        if parametrization == 'Gaussian':
            # for Gaussian, param1 == mu, param2 == sigma
            epsilon = param2.data.new(param2.size()).normal_()
            out = param1 + param2 * epsilon

        elif parametrization == 'Kumaraswamy':
            # for Kumaraswamy, param1 == alpha, param2 == beta
            v = self.set_v_K_to_one(self.get_kumaraswamy_samples(param1, param2))
            out = self.get_stick_segments(v)

        elif parametrization == 'GEM':
            # for GEM, param1 == alpha, param2 == beta
            v = self.set_v_K_to_one(self.get_GEM_samples(param1, param2))
            out = self.get_stick_segments(v)

        elif parametrization == 'Gauss_Logit':
            # for Gauss-Logit, param1 == mu, param2 == sigma
            epsilon = param2.data.new(param2.size()).normal_()
            v = self.set_v_K_to_one(logistic_func(param1 + param2 * epsilon))
            out = self.get_stick_segments(v)

        return out

    def get_kumaraswamy_samples(self, param1, param2):
        # u is analogous to epsilon noise term in the Gaussian VAE
        u = self.uniform_distribution.sample([1]).squeeze()
        v = (1 - u.pow(1 / param2)).pow(1 / param1)
        return v  # sampled fractions

    def get_GEM_samples(self, param1, param2):
        u_hat = self.uniform_distribution.sample([1]).squeeze()
        v = (u_hat * param1 * torch.lgamma(param1).exp()).pow(1 / param1) / param2

        # poor_approx_idx = torch.where((param1 >= 1) * (1 - .94*u_hat*param1.log() >= -.42)) # Kingma & Welling (2014)
        # poor_approx_idx = torch.where(param1 >= 1)
        # if poor_approx_idx[0].nelement() != 0:
        #     v1 = param1 / (param1 + param2)
        #     v[poor_approx_idx] = v1[poor_approx_idx]

        return v

    def set_v_K_to_one(self, v):
        # set Kth fraction v_i,K to one to ensure the stick segments sum to one
        if v.ndim > 2:
            v = v.squeeze()
        v0 = v[:, -1].pow(0).reshape(v.shape[0], 1)
        v1 = torch.cat([v[:, :latent_ndims - 1], v0], dim=1)
        return v1

    def get_stick_segments(self, v):
        n_samples = v.size()[0]
        n_dims = v.size()[1]
        pi = torch.zeros((n_samples, n_dims))

        for k in range(n_dims):
            if k == 0:
                pi[:, k] = v[:, k]
            else:
                pi[:, k] = v[:, k] * torch.stack([(1 - v[:, j]) for j in range(n_dims) if j < k]).prod(axis=0)

        # ensure stick segments sum to 1
        assert_almost_equal(torch.ones(n_samples), pi.sum(axis=1).detach().numpy(),
                            decimal=2, err_msg='stick segments do not sum to 1')
        return pi


class GaussianVAE(torch.nn.Module, GaussianEncoder, Decoder, VAE):
    def __init__(self):
        super(GaussianVAE, self).__init__()
        GaussianEncoder.__init__(self)
        Decoder.__init__(self)
        VAE.__init__(self, target_distribution=torch.distributions.MultivariateNormal,
                     latent_distribution=torch.distributions.MultivariateNormal,
                     prior_param1=torch.ones(latent_ndims) * prior_mu,
                     prior_param2=torch.diag(torch.ones(latent_ndims) * prior_sigma ** 2))
        self.parametrization = 'Gaussian'

    def forward(self, x):
        mu, sigma = self.encode(x.view(-1, input_ndims))
        z = self.reparametrize(mu, sigma, parametrization='Gaussian') if self.training else mu
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, torch.stack([torch.diag(sigma[i]).pow(2) for i in range(len(sigma))])

    def kl_divergence(self, mu, sigma):
        q = self.latent_distribution(mu, sigma)
        p = self.target_distribution(self.prior_param1, self.prior_param2)
        kl = torch.distributions.kl_divergence(q, p)
        return kl


class StickBreakingVAE(torch.nn.Module, StickBreakingEncoder, Decoder, VAE):
    def __init__(self, parametrization):
        super(StickBreakingVAE, self).__init__()
        StickBreakingEncoder.__init__(self)
        Decoder.__init__(self)
        self.parametrization = parametrization

        if parametrization == 'Kumaraswamy':
            VAE.__init__(self, target_distribution=torch.distributions.beta.Beta,
                         latent_distribution=torch.distributions.kumaraswamy.Kumaraswamy,
                         prior_param1=torch.ones(latent_ndims) * prior_alpha,
                         prior_param2=torch.ones(latent_ndims) * prior_beta)
        elif parametrization == 'Gauss_Logit':
            VAE.__init__(self, target_distribution=torch.distributions.MultivariateNormal,
                         latent_distribution=torch.distributions.MultivariateNormal,
                         prior_param1=torch.ones(latent_ndims) * glogit_prior_mu,
                         prior_param2=torch.diag(torch.ones(latent_ndims) * prior_sigma ** 2))
        elif parametrization == 'GEM':
            # Gamma distribution used to approximate beta distribution
            VAE.__init__(self, target_distribution=torch.distributions.gamma.Gamma,
                         latent_distribution=torch.distributions.gamma.Gamma,
                         prior_param1=torch.ones(latent_ndims) * prior_alpha,
                         prior_param2=torch.ones(latent_ndims) * prior_beta)

    def forward(self, x):
        param1, param2 = self.encode(x.view(-1, input_ndims))
        if self.training:
            pi = self.reparametrize(param1, param2, parametrization=self.parametrization)
        else:
            # reconstruct random samples from the area of highest density
            if self.parametrization == 'Kumaraswamy':
                highest_density = (1 - self.latent_distribution(param1, param2).mean.pow(1 / param2)).pow(1 / param1)
            elif self.parametrization == 'GEM':
                highest_density = torch.lgamma(param1).exp().mul(param1).\
                    mul(torch.distributions.Beta(param1, param2).mean).pow(1 / param1).div(param2)
            elif self.parametrization == 'Gauss_Logit':
                highest_density = param1

            v = self.set_v_K_to_one(highest_density)
            pi = self.get_stick_segments(v)

        reconstructed_x = self.decode(pi)

        if self.parametrization == 'Gauss_Logit':
            param2 = torch.stack([torch.diag(param2[i].pow(2)) for i in range(len(param2))])

        return reconstructed_x, param1, param2

    def kl_divergence(self, param1, param2):
        kl_switcher = dict(Kumaraswamy=self.kumaraswamy_kl_divergence,
                           GEM=self.gamma_kl_divergence,
                           Gauss_Logit=self.gauss_logit_kl_divergence)
        kl_divergence_func = kl_switcher.get(self.parametrization)

        assert((param1 != 0).all(), f'Zero at alpha indices: {torch.nonzero((param1!=0) == False, as_tuple=False).squeeze()}')
        assert((param2 != 0).all(), f'Zero at beta indices: {torch.nonzero((param2!=0) == False, as_tuple=False).squeeze()}')

        return kl_divergence_func(param1, param2)

    def gauss_logit_kl_divergence(self, mu, sigma):
        q = self.latent_distribution(mu, sigma)  # note: KL term should be between normal & normal
        p = self.target_distribution(self.prior_param1, self.prior_param2)
        kl = torch.distributions.kl_divergence(q, p)
        return kl

    def gamma_kl_divergence(self, alpha, beta):
        q1 = self.latent_distribution(alpha, 1)
        q2 = self.latent_distribution(beta, 1)
        p1 = self.target_distribution(self.prior_param1, 1)
        p2 = self.target_distribution(self.prior_param2, 1)
        kl1 = torch.distributions.kl_divergence(q1, p1)
        kl2 = torch.distributions.kl_divergence(q2, p2)
        return (kl1 + kl2).sum()

    def kumaraswamy_kl_divergence(self, alpha, beta):
        psi_b_taylor_approx = beta.log() - 1. / beta.mul(2) - 1. / beta.pow(2).mul(12)
        kl = ((alpha - prior_alpha) / alpha) * (-0.57721 - psi_b_taylor_approx - 1 / beta)
        kl += alpha.mul(beta).log() + beta_func(prior_alpha, prior_beta).log()  # normalization constants
        kl += - (beta - 1) / beta
        kl += torch.stack([1. / (i + alpha * beta) * beta_func(i / alpha, beta) for i in range(1, 11)]).sum(axis=0) \
              * (prior_beta - 1) * beta  # 10th-order Taylor approximation

        return kl.sum()

class USDN(torch.nn.Module, StickBreakingEncoderMSI, StickBreakingEncoderHSI, Decoder):
    def __init__(self, msi_dim, hsi_dim, out_dim, parametrization):
        super(USDN, self).__init__()
        StickBreakingEncoderMSI.__init__(self, msi_dim)
        StickBreakingEncoderHSI.__init__(self, hsi_dim)
        Decoder.__init__(self, out_dim)
        
        
        self.parametrization = parametrization
        if parametrization == 'Kumaraswamy':
            VAE.__init__(self, target_distribution=torch.distributions.beta.Beta,
                         latent_distribution=torch.distributions.kumaraswamy.Kumaraswamy,
                         prior_params=[torch.ones(latent_ndims) * prior_alpha,
                         torch.ones(latent_ndims) * prior_beta])
        elif parametrization == 'Gauss_Logit':
            VAE.__init__(self, target_distribution=torch.distributions.MultivariateNormal,
                         latent_distribution=torch.distributions.MultivariateNormal,
                         prior_params[torch.ones(latent_ndims) * glogit_prior_mu,
                         torch.diag(torch.ones(latent_ndims) * prior_sigma ** 2)])
        elif parametrization == 'GEM':
            # Gamma distribution used to approximate beta distribution
            VAE.__init__(self, target_distribution=torch.distributions.gamma.Gamma,
                         latent_distribution=torch.distributions.gamma.Gamma,
                         prior_params=[torch.ones(latent_ndims) * prior_alpha,
                         torch.ones(latent_ndims) * prior_beta])

        # self.init_weights(self.msi_encoder.modules())
        # self.init_weights(self.hsi_encoder.modules())

    def forward(self, msi_image, hsi_image):
        pdb.set_trace()
        param1, param2 = self.encode(msi_image.view(-1, input_ndims))
        param3, param4 = self.encode(hsi_image.view(-1, input_ndims))
        if self.training:
            pi_m = self.reparametrize(param1, param2, parametrization=self.parametrization)
            pi_h = self.reparametrize(param3, param4, parametrization=self.parametrization)
        else:
            # reconstruct random samples from the area of highest density
            if self.parametrization == 'Kumaraswamy':
                highest_density_msi = (1 - self.latent_distribution(param1, param2).mean.pow(1 / param2)).pow(1 / param1)
                highest_density_hsi = (1 - self.latent_distribution(param3, param4).mean.pow(1 / param4)).pow(1 / param3)
            elif self.parametrization == 'GEM':
                highest_density = torch.lgamma(param1).exp().mul(param1).\
                    mul(torch.distributions.Beta(param1, param2).mean).pow(1 / param1).div(param2)
            elif self.parametrization == 'Gauss_Logit':
                highest_density = param1

            v_m = self.set_v_K_to_one(highest_density_msi)
            pi_m = self.get_stick_segments(v_m)
            v_h = self.set_v_K_to_one(highest_density_hsi)
            pi_h = self.get_stick_segments(v_h)
            # changes
        reconstructed_msi = self.decode(pi_m)
        reconstructed_hsi = self.decode(pi_h)

        if self.parametrization == 'Gauss_Logit':
            param2 = torch.stack([torch.diag(param2[i].pow(2)) for i in range(len(param2))])

        return reconstructed_x, param1, param2, param3, param4

    def kl_divergence(self, param1, param2):
        kl_switcher = dict(Kumaraswamy=self.kumaraswamy_kl_divergence,
                           GEM=self.gamma_kl_divergence,
                           Gauss_Logit=self.gauss_logit_kl_divergence)
        kl_divergence_func = kl_switcher.get(self.parametrization)

        assert((param1 != 0).all(), f'Zero at alpha indices: {torch.nonzero((param1!=0) == False, as_tuple=False).squeeze()}')
        assert((param2 != 0).all(), f'Zero at beta indices: {torch.nonzero((param2!=0) == False, as_tuple=False).squeeze()}')

        return kl_divergence_func(param1, param2)

    def gauss_logit_kl_divergence(self, mu, sigma):
        q = self.latent_distribution(mu, sigma)  # note: KL term should be between normal & normal
        p = self.target_distribution(self.prior_param1, self.prior_param2)
        kl = torch.distributions.kl_divergence(q, p)
        return kl

    def gamma_kl_divergence(self, alpha, beta):
        q1 = self.latent_distribution(alpha, 1)
        q2 = self.latent_distribution(beta, 1)
        p1 = self.target_distribution(self.prior_param1, 1)
        p2 = self.target_distribution(self.prior_param2, 1)
        kl1 = torch.distributions.kl_divergence(q1, p1)
        kl2 = torch.distributions.kl_divergence(q2, p2)
        return (kl1 + kl2).sum()

    def kumaraswamy_kl_divergence(self, alpha, beta):
        psi_b_taylor_approx = beta.log() - 1. / beta.mul(2) - 1. / beta.pow(2).mul(12)
        kl = ((alpha - prior_alpha) / alpha) * (-0.57721 - psi_b_taylor_approx - 1 / beta)
        kl += alpha.mul(beta).log() + beta_func(prior_alpha, prior_beta).log()  # normalization constants
        kl += - (beta - 1) / beta
        kl += torch.stack([1. / (i + alpha * beta) * beta_func(i / alpha, beta) for i in range(1, 11)]).sum(axis=0) \
              * (prior_beta - 1) * beta  # 10th-order Taylor approximation

        return kl.sum()
