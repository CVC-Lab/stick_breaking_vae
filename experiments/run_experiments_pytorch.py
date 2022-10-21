import datetime
import os
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.util_vars import (CUDA, learning_rate, print_interval, n_train_epochs,
    latent_ndims, parametrizations, lookahead, n_monte_carlo_samples, 
    n_random_samples, model_path, 
    checkpoint_path, hr_msi_ndims,lr_hsi_ndims, hr_hsi_ndims, 
    hr_hsi_shape, train_loader, test_loader, R)
from utils.util_funcs import image_grid, plot_to_image
from model_classes.VAEs_pytorch import GaussianVAE, StickBreakingVAE, USDN
# import wandb
import pdb

# wandb.init(project="sharp_vae", sync_tensorboard=True)
# init model and optimizer
time_now = datetime.datetime.now().__format__('%b_%d_%Y_%H_%M')
parametrization = parametrizations['Kumar']
# model = GaussianVAE().cuda() if CUDA else GaussianVAE()
# model = StickBreakingVAE(parametrization).cuda() if CUDA else StickBreakingVAE(parametrization)
model = USDN(hr_msi_ndims, lr_hsi_ndims, hr_hsi_ndims, R, parametrization)
# wandb.watch(model)
optimizer = optim.Adam(model.parameters(), betas=(0.95, 0.999), lr=learning_rate, eps=1e-4, weight_decay=1e-3)
parametrization_str = parametrization if model._get_name() == "StickBreakingVAE" else ''
model_name = '_'.join(filter(None, [model._get_name(), parametrization_str]))
start_epoch = 1

if checkpoint_path:  # load checkpoint state
    assert model_name in checkpoint_path, 'Mismatch between specified and checkpoint parametrization'
    checkpoint = torch.load(checkpoint_path)
    start_epoch, model_state_dict, optimizer_state_dict = list(checkpoint.values())
    optimizer.load_state_dict(optimizer_state_dict)
    model.load_state_dict(model_state_dict)

# init save directories
tb_writer = SummaryWriter(f'logs/{model_name}')
if not os.path.exists(os.path.join(model_path, model_name)):
    os.mkdir(os.path.join(model_path, model_name))

best_test_epoch = None
best_test_loss = None
best_test_model = None
best_test_optimizer = None
stop_training = None

stages = {
    0:"train_hyperspectral",
    1:"train_multispectral",
    2:"minimize_angular_distortion",
    3: "eval"
}

def freeze_network(params):
    for param in params:
        param.require_grad = False

def unfreeze_network(params):
    for param in params:
        param.require_grad = True

def train(epoch):
    print(f'\nTrain Epoch: {epoch}')
    model.train()
    reconstruction_loss = 0
    regularization_loss = 0

    for batch_idx, data in enumerate(train_loader):
        
        data = data.cuda() if CUDA else data
        optimizer.zero_grad()
        # STAGE 1
        # freeze msi encoder, train HSI network
        freeze_network(model.msi_encoder_layers)
        # if epoch % 6 ==0 and epoch > 0:
        #     pdb.set_trace()
        recon_hsi_batch, param1_hsi, param2_hsi, _ = model(data, stage=0)
        batch_hsi_reconstruction_loss, batch_hsi_regularization_loss = \
            model.ELBO_loss(recon_hsi_batch, data[2], hr_hsi_ndims, param1_hsi, param2_hsi, model.kl_divergence)
        reconstruction_loss += batch_hsi_reconstruction_loss
        regularization_loss += batch_hsi_regularization_loss
        loss = batch_hsi_reconstruction_loss + batch_hsi_regularization_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # to prevent exploding gradient
        # check for NaN values in all .grad attributes
        for name, param in model.named_parameters():
            isfinite = torch.isfinite(param).all()
            if not isfinite:
                print(f"Nan in stage: {1}")
                print(name, isfinite)
                torch.save({'epoch': best_test_epoch,
                            'model_state_dict': best_test_model,
                            'optimizer_state_dict': best_test_optimizer},
                           os.path.join(model_path, model_name, f'finite_loss_checkpoint_{model_name}_{time_now}'))
                break
        optimizer.step()

        # STAGE 2
        # freeze hsi network, train msi encoder
        optimizer.zero_grad()
        unfreeze_network(model.msi_encoder_layers)
        freeze_network(model.hsi_encoder_layers)

        recon_msi_batch, param1_msi, param2_msi, _ = model(data, stage=1)
        batch_msi_reconstruction_loss, batch_msi_regularization_loss = \
            model.ELBO_loss(recon_msi_batch, data[2], hr_hsi_ndims, param1_msi, param2_msi, model.kl_divergence)
        reconstruction_loss += batch_msi_reconstruction_loss
        regularization_loss += batch_msi_regularization_loss
        loss = batch_msi_reconstruction_loss + batch_msi_regularization_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # to prevent exploding gradient
        # check for NaN values in all .grad attributes
        for name, param in model.named_parameters():
            
            isfinite = torch.isfinite(param).all()
            if not isfinite:
                print(f"Nan in stage: {2}")
                print(name, isfinite)
                torch.save({'epoch': best_test_epoch,
                            'model_state_dict': best_test_model,
                            'optimizer_state_dict': best_test_optimizer},
                           os.path.join(model_path, model_name, f'finite_loss_checkpoint_{model_name}_{time_now}'))
                break
        optimizer.step()

        # STEP 3
        # hsi network is froze, only msi encoder is updated
        optimizer.zero_grad()
        if epoch % 40 == 0 and epoch > 0:
            # again call model data and get Sm and Sh and then minimize loss
            Sm, Sh = model(data, stage=2)
            spectral_distortion_loss = model.sam_loss(Sm, Sh)
            spectral_distortion_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            for name, param in model.named_parameters():
                isfinite = torch.isfinite(param).all()
                if not isfinite:
                    print(f"Nan in stage: {3}")
                    print(name, isfinite)
                    torch.save({'epoch': best_test_epoch,
                                'model_state_dict': best_test_model,
                                'optimizer_state_dict': best_test_optimizer},
                            os.path.join(model_path, model_name, f'finite_loss_checkpoint_{model_name}_{time_now}'))
                    break
            optimizer.step()
            # pdb.set_trace()
        unfreeze_network(model.hsi_encoder_layers)

        if batch_idx % print_interval == 0:
            print(f'[{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item() / len(data):.6f}')

    train_loss = reconstruction_loss + regularization_loss

    reconstruction_loss /= len(train_loader.dataset)
    regularization_loss /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))

    tb_writer.add_scalar(f"{time_now}/Loss/train", train_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Regularization_Loss/train", regularization_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Reconstruction_Loss/train", reconstruction_loss.item(), epoch)


def test(epoch):
    global best_test_epoch, best_test_loss, stop_training
    model.eval()
    model.training = False
    reconstruction_loss = 0
    regularization_loss = 0

    for batch_idx, data in enumerate(test_loader):
        data = data.cuda() if CUDA else data
        
        recon_hsi_batch, param1_hsi, param2_hsi, Sh = model(data, stage=3)
        batch_hsi_reconstruction_loss, batch_hsi_regularization_loss = \
            model.ELBO_loss(recon_hsi_batch, data[2], hr_hsi_ndims, param1_hsi, param2_hsi, model.kl_divergence)
        reconstruction_loss += batch_hsi_reconstruction_loss
        regularization_loss += batch_hsi_regularization_loss
        loss = batch_hsi_reconstruction_loss + batch_hsi_regularization_loss

    test_loss = reconstruction_loss + regularization_loss

    reconstruction_loss /= len(test_loader.dataset)
    regularization_loss /= len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    tb_writer.add_scalar(f"{time_now}/Loss/test", test_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Regularization_Loss/test", regularization_loss.item(), epoch)
    tb_writer.add_scalar(f"{time_now}/Reconstruction_Loss/test", reconstruction_loss.item(), epoch)

    if epoch == start_epoch:
        best_test_epoch = epoch
        best_test_loss = test_loss
    else:
        best_test_epoch = epoch if test_loss < best_test_loss else best_test_epoch
        best_test_loss = test_loss if best_test_epoch else best_test_loss
        stop_training = True if epoch - best_test_epoch > lookahead else False


for epoch in range(start_epoch, n_train_epochs + 1):
    try:
        train(epoch)
        test(epoch)
        model.eval()
    except AssertionError:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(model_path, model_name, f'failed_checkpoint_{model_name}_{time_now}'))
        print('Stick segments do not sum to one/reconstructed_x.log() is not finite. Stopping training.')
        break

    if epoch == best_test_epoch and epoch % 5 == 0:
        
        sample = torch.randn(n_random_samples, latent_ndims)  # TODO: normalize random sample to latent space bounds
        sample = sample.cuda() if CUDA else sample
        sample = model.decode(sample).cpu()
        
        # save latent space samples
        img_tensor = sample.view(n_random_samples, *hr_hsi_shape)
        for sid in range(n_random_samples):
            figure = image_grid(img_tensor[sid, ...], sid)
            tb_writer.add_image(f'{sid}_random_latent_space_samples_{time_now}',
            img_tensor=plot_to_image(figure)[..., :3],
            global_step=epoch,
            dataformats='HWC')

        # save originals  
        random_idxs = torch.randint(0, len(test_loader.dataset), size=(n_random_samples,))
        samples = [test_loader.dataset[idx] for idx in random_idxs]
        img_tensor = sample.view(n_random_samples, *hr_hsi_shape)
        for sid in range(n_random_samples):
            figure = image_grid(img_tensor[sid, ...], sid)
            tb_writer.add_image(f'{sid}_original_test_samples_{time_now}',
            img_tensor=plot_to_image(figure)[..., :3],
            global_step=epoch,
            dataformats='HWC')

        samples = samples.cuda() if CUDA else samples

        samples = torch.stack([model((x[0][None, ...], x[1][None, ...], 
                                        x[2][None, ...]), stage=3)[0] for x in samples])
        # save reconstructed
        img_tensor = sample.view(n_random_samples, *hr_hsi_shape)
        
        for sid in range(n_random_samples):
            figure = image_grid(img_tensor[sid, ...], sid)
            tb_writer.add_image(f'{sid}_reconstructed_test_samples_{time_now}',
            img_tensor=plot_to_image(figure)[..., :3],
            global_step=epoch,
            dataformats='HWC')

        # save trained weights
        best_test_model = model.state_dict().copy()
        best_test_optimizer = optimizer.state_dict().copy()

    elif stop_training:
        break

tb_writer.close()
torch.save({'epoch': best_test_epoch,
            'model_state_dict': best_test_model,
            'optimizer_state_dict': best_test_optimizer},
           os.path.join(model_path, model_name, f'best_checkpoint_{model_name}_{time_now}'))


