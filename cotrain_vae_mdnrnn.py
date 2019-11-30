import argparse
from collections import namedtuple
from functools import partial
from os.path import join, exists
from os import mkdir

import numpy as np
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutSequenceDataset

parser = argparse.ArgumentParser(description='VAE & MDNRNN Co-Trainer')
parser.add_argument('--originallogdir', type=str, help='Directory where results were logged from pretraining')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--datasets', type=str, default='datasets',
                    help='Where the datasets are stored')
parser.add_argument('--sequence_length', type=int, default=32)

parser.add_argument('--test_only', type=bool, default=False)
args = parser.parse_args()

# PyTorch setup
cuda = torch.cuda.is_available()
print('CUDA: {}'.format(cuda))
torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

# constants
BSIZE = 16
SEQ_LEN = args.sequence_length
epochs = 200

# Data Loading
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset(join(args.datasets, 'carracing'), SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset(join(args.datasets, 'carracing'), SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=8)

# Load VAE
vae_file = join(args.originallogdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the originallogdir..."
state = torch.load(vae_file, map_location={'cuda:0': str(device)})
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))
vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])
vae_optimizer = optim.Adam(vae.parameters())
vae_scheduler = ReduceLROnPlateau(vae_optimizer, 'min', factor=0.5, patience=5)

# Load RNN
rnn_dir = join(args.originallogdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')
assert exists(rnn_file), 'No trained MDNRNN in the originallogdir...'
mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
mdrnn_optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
mdrnn_scheduler = ReduceLROnPlateau(mdrnn_optimizer, 'min', factor=0.5, patience=5)

rnn_state = torch.load(rnn_file, map_location={'cuda:0': str(device)})
print("Loading MDRNN at epoch {} "
      "with test error {}".format(
          rnn_state["epoch"], rnn_state["precision"]))
mdrnn.load_state_dict(rnn_state["state_dict"])

VaeLoss = namedtuple(
    'VaeLoss',
    ['reconstruction', 'kld', 'reward', 'death', 'loss'])

def vae_loss_function(
    recon_x, x, mu, logsigma, reward, predicted_reward, 
    death, predicted_death):
    """ VAE loss function """
    # print('recon_x', recon_x.shape)
    # print('x', x.shape)
    # print('mu', mu.shape)
    # print('logsigma', logsigma.shape)
    # print('reward', reward.shape)
    # print('predicted_reward', predicted_reward.shape)
    # print('death', death.shape)
    # print('predicted_death', predicted_death.shape)
    reconstruction = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = (
        -0.5 * torch.sum(
            1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp()))
    kld /= np.prod(x.size())
    
    reward = F.binary_cross_entropy_with_logits(predicted_reward, reward)

    death = F.binary_cross_entropy_with_logits(predicted_death, death)

    loss = reconstruction + kld + reward + death

    return VaeLoss(
        reconstruction=reconstruction, kld=kld,
        reward=reward, death=death, loss=loss)


MdnRnnPrediction = namedtuple(
    'MdnRnnPrediction',
    ['mus', 'sigmas', 'logpi', 'rs', 'ds'])

def get_mdn_rnn_prediction(
    latent_obs, action, reward, terminal, latent_next_obs):
    latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
    return MdnRnnPrediction(*mdrnn(action, latent_obs))

def mdn_rnn_loss_function(
    latent_obs, action, reward, terminal, latent_next_obs,
    mdn_rnn_prediction):    
    latent_obs, action,\
        reward, terminal,\
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
    mus, sigmas, logpi, rs, ds = mdn_rnn_prediction
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = F.binary_cross_entropy_with_logits(ds, terminal)
    mse = F.binary_cross_entropy_with_logits(rs, reward)
    scale = LSIZE + 2
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

VaeLatentRepresentation = namedtuple(
    'VaeLatentRepresentation',
    ['latent_obs', 'recon_obs', 'obs_mu', 'obs_logsigma'])

def reshape_and_upsample_obs(obs):
    return F.upsample(
        obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
        mode='bilinear', align_corners=True)

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    obs, next_obs = [
        reshape_and_upsample_obs(x)
        for x in (obs, next_obs)]

    (recon_obs, obs_mu, obs_logsigma), (recon_next_obs, next_obs_mu, next_obs_logsigma) = [
        vae(x) for x in (obs, next_obs)]

    latent_obs, latent_next_obs = [
        (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(BSIZE, SEQ_LEN, LSIZE)
        for x_mu, x_logsigma in
        [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return (
        VaeLatentRepresentation(
            latent_obs=latent_obs,
            recon_obs=recon_obs,
            obs_mu=obs_mu,
            obs_logsigma=obs_logsigma),
        VaeLatentRepresentation(
            latent_obs=latent_next_obs,
            recon_obs=recon_next_obs,
            obs_mu=next_obs_mu,
            obs_logsigma=next_obs_logsigma),)

def data_pass(epoch, train):
    if train:
        mdrnn.train()
        vae.train()
        loader = train_loader
    else:
        mdrnn.eval()
        vae.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_mdn_rnn_loss = 0
    cum_mdn_rnn_gmm = 0
    cum_mdn_rnn_bce = 0
    cum_mdn_rnn_mse = 0
    cum_vae_loss = 0
    cum_vae_rec_loss = 0
    cum_vae_kld_loss = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]
        reward = (reward.sign() == 1).double()

        latent_rep, latent_next_rep = to_latent(obs, next_obs)
        mdn_rnn_prediction = get_mdn_rnn_prediction(
            latent_rep.latent_obs, action, reward, terminal,
            latent_next_rep.latent_obs)
        get_vae_loss = lambda: vae_loss_function(
            latent_rep.recon_obs, reshape_and_upsample_obs(obs),
            latent_rep.obs_mu, latent_rep.obs_logsigma,
            reward, mdn_rnn_prediction.rs.transpose(1, 0),
            terminal, mdn_rnn_prediction.ds.transpose(1, 0))
        get_mdn_rnn_loss = lambda: mdn_rnn_loss_function(
            latent_rep.latent_obs, action, reward, terminal,
            latent_next_rep.latent_obs, mdn_rnn_prediction)
        if train:
            vae_optimizer.zero_grad()
            vae_loss = get_vae_loss()
            vae_loss.loss.backward(retain_graph=True)
            vae_optimizer.step()
            mdrnn_optimizer.zero_grad()
            mdn_rnn_loss = get_mdn_rnn_loss()
            mdn_rnn_loss['loss'].backward()
            mdrnn_optimizer.step()
        else:
            with torch.no_grad():
                vae_loss = get_vae_loss()
                mdn_rnn_loss = get_mdn_rnn_loss()

        cum_mdn_rnn_loss += mdn_rnn_loss['loss'].item()
        cum_mdn_rnn_gmm += mdn_rnn_loss['gmm'].item()
        cum_mdn_rnn_bce += mdn_rnn_loss['bce'].item()
        cum_mdn_rnn_mse += mdn_rnn_loss['mse'].item()
        cum_vae_loss += vae_loss.loss.item()
        cum_vae_rec_loss += vae_loss.reconstruction.item()
        cum_vae_kld_loss += vae_loss.kld.item()

        pbar.set_postfix_str(
            "mdn_loss={mdn_loss:10.6f} bce={bce:10.6f} "
            "gmm={gmm:10.6f} mse={mse:10.6f} "
            "vae_loss={vae_loss:10.6f} rec={rec:10.6f} "
            "kld={kld:10.6f}".format(
                mdn_loss=cum_mdn_rnn_loss / (i + 1), 
                bce=cum_mdn_rnn_bce / (i + 1),
                gmm=cum_mdn_rnn_gmm / LSIZE / (i + 1),
                mse=cum_mdn_rnn_mse / (i + 1),
                vae_loss=cum_vae_loss / (i + 1),
                rec=cum_vae_rec_loss / (i + 1),
                kld=cum_vae_kld_loss / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return (
        cum_mdn_rnn_loss * BSIZE / len(loader.dataset),
        cum_vae_loss / len(loader.dataset))


train = partial(data_pass, train=True)
test = partial(data_pass, train=False)

new_vae_dir = join(args.logdir, 'vae')
if not exists(new_vae_dir):
    mkdir(new_vae_dir)

new_rnn_dir = join(args.logdir, 'mdrnn')
if not exists(new_rnn_dir):
    mkdir(new_rnn_dir)

new_samples_dir = join(args.logdir, 'vae', 'samples')
if not exists(new_samples_dir):
    mkdir(new_samples_dir)

if args.test_only:
    test(e)

for e in range(epochs):
    train(e)
    mdn_rnn_test_loss, vae_test_loss = test(e)
    vae_scheduler.step(vae_test_loss)
    mdrnn_scheduler.step(mdn_rnn_test_loss)

    vae_checkpoint_fname = join(
        new_vae_dir, 'checkpoint_{e}.tar'.format(e=e))
    mdn_rnn_checkpoint_fname = join(
        new_rnn_dir, 'checkpoint_{e}.tar'.format(e=e))
    torch.save({
        'epoch': e,
        'state_dict': vae.state_dict(),
        'precision': vae_test_loss,
        'optimizer': vae_optimizer.state_dict(),
        'scheduler': vae_scheduler.state_dict(),
        }, vae_checkpoint_fname)
    torch.save({
        'state_dict': mdrnn.state_dict(),
        'optimizer': mdrnn_optimizer.state_dict(),
        'scheduler': mdrnn_scheduler.state_dict(),
        'precision': mdn_rnn_test_loss,
        'epoch': e
        }, mdn_rnn_checkpoint_fname)

    with torch.no_grad():
        sample = torch.randn(RED_SIZE, LSIZE).to(device)
        sample = vae.decoder(sample).cpu()
        save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                   join(new_samples_dir, 'sample_' + str(e) + '.png'))



