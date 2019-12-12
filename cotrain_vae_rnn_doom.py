""" Recurrent model training """
import argparse
from collections import namedtuple
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import RED_SIZE, SIZE
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau

from data.loaders import VariableLengthRolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss

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
ASIZE = 1
BSIZE = 16
SEQ_LEN = 140  # 4 seconds
LSIZE = 64
RSIZE = 512
epochs = 200

# Load VAE
vae_file = join(args.originallogdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the originallogdir..."
state = torch.load(vae_file, map_location={'cuda:0': str(device)})
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))
vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])
vae_optimizer = torch.optim.Adam(vae.parameters())
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


def collation_fn(rollouts): 
    rollout_items = [ 
        [], [], [], [], []] 
    for rollout in rollouts: 
        for i in range(len(rollout)): 
            rollout_items[i].append(torch.Tensor(rollout[i])) 
    for i in range(len(rollout_items)): 
        rollout_items[i] = pack_sequence(sorted(rollout_items[i], key=len, reverse=True)) 
    return tuple(rollout_items)

# Data Loading
transform = lambda x: np.transpose(x, (0, 3, 1, 2)) / 255
train_loader = DataLoader(
    VariableLengthRolloutSequenceDataset(join(args.datasets#, 'doom'
        ), SEQ_LEN, transform=transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, collate_fn=collation_fn)
test_loader = DataLoader(
    VariableLengthRolloutSequenceDataset(join(args.datasets#, 'doom'
        ), SEQ_LEN, transform=transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=8, collate_fn=collation_fn)

VaeLatentRepresentation = namedtuple(
    'VaeLatentRepresentation',
    ['latent_obs', 'recon_obs', 'obs_mu', 'obs_logsigma'])

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """

    (obs, lengths), (next_obs, _) = [pad_packed_sequence(o) for o in (obs, next_obs,)]
    max_length, batch_size = obs.shape[:2]
    (recon_obs, obs_mu, obs_logsigma), (next_recon_obs, next_obs_mu, next_obs_logsigma) = [
        [y.reshape((max_length, batch_size, LSIZE)) 
         if i != 0 else y.reshape((max_length, batch_size, 3, SIZE, SIZE))
         for i, y in enumerate(vae(x.reshape((-1, 3, SIZE, SIZE))))]
        for x in (obs, next_obs)]

    latent_obs, latent_next_obs = [
        (x_mu + x_logsigma.exp() * torch.randn_like(x_mu))
        for x_mu, x_logsigma in
        [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

    return (
        VaeLatentRepresentation(
            latent_obs=pack_padded_sequence(latent_obs, lengths),
            recon_obs=pack_padded_sequence(recon_obs, lengths),
            obs_mu=pack_padded_sequence(obs_mu, lengths),
            obs_logsigma=pack_padded_sequence(obs_logsigma, lengths)),
        VaeLatentRepresentation(
            latent_obs=pack_padded_sequence(latent_next_obs, lengths),
            recon_obs=pack_padded_sequence(next_recon_obs, lengths),
            obs_mu=pack_padded_sequence(next_obs_mu, lengths),
            obs_logsigma=pack_padded_sequence(next_obs_logsigma, lengths)))

MdnRnnPrediction = namedtuple(
    'MdnRnnPrediction',
    ['mus', 'sigmas', 'logpi', 'rs', 'ds'])

def get_mdn_rnn_prediction(
    latent_obs, action):
    (latent_obs, lengths), (action, _) = [
        pad_packed_sequence(a) for a in [latent_obs, action]
    ]
    return MdnRnnPrediction(*[
        pack_padded_sequence(p, lengths)
        for p in mdrnn(action.unsqueeze(-1), latent_obs)])

VaeLoss = namedtuple(
    'VaeLoss',
    ['reconstruction', 'kld', 'death', 'loss'])

def vae_loss_function(
    recon_x, x, mu, logsigma, death, predicted_death):
    (recon_x, lengths), (x, _), (mu, _), (logsigma, _), (death, _), (predicted_death, _) = [
        pad_packed_sequence(i) for i in [
            recon_x, x, mu, logsigma, death, predicted_death
        ]
    ]

    reconstruction_losses = []
    kld_losses = []
    death_losses = []
    for b, length in enumerate(lengths):
        reconstruction_losses.append(F.mse_loss(
            recon_x[:length, b], x[:length, b]))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_losses.append(
            -0.5 * torch.sum(
                1 
                + 2 * logsigma[:length, b] 
                - mu[:length, b].pow(2) 
                - (2 * logsigma[:length, b]).exp())
            / np.prod(x[:length, b].size()))
        
        death_losses.append(
            F.binary_cross_entropy_with_logits(
                predicted_death[:length, b], death[:length, b]))

    reconstruction = torch.mean(torch.stack(reconstruction_losses))
    kld = torch.mean(torch.stack(kld_losses))
    death = torch.mean(torch.stack(death_losses))

    loss = reconstruction + kld + death

    return VaeLoss(
        reconstruction=reconstruction,
        kld=kld,
        death=death,
        loss=loss)


def mdn_rnn_loss_function(
    latent_obs, action, reward, terminal, latent_next_obs,
    mdn_rnn_prediction):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    (latent_obs, lengths), (action, _),\
        (reward, _), (terminal, _),\
        (latent_next_obs, _) = [
        pad_packed_sequence(a)
        for a in [latent_obs, action,
                    reward, terminal,
                    latent_next_obs]]
    (mus, _), (sigmas, _), (logpi, _), (rs, _), (ds, _) = [
        pad_packed_sequence(o) for o in mdn_rnn_prediction
    ]
    gmm_losses = []
    bce_losses = []
    for b, length in enumerate(lengths):
        gmm_losses.append(
            gmm_loss(latent_next_obs[:length, b], mus[:length, b], sigmas[:length, b], logpi[:length, b]))
        bce_losses.append(
            F.binary_cross_entropy_with_logits(ds[:length, b], terminal[:length, b]))
    gmm = torch.mean(torch.stack(gmm_losses))
    bce = torch.mean(torch.stack(bce_losses))
    loss = (gmm + bce) / (LSIZE + 1)
    return dict(gmm=gmm, bce=bce, loss=loss)


def data_pass(epoch, train): # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
    else:
        mdrnn.eval()
        loader = test_loader

    cum_mdn_rnn_loss = 0
    cum_mdn_rnn_gmm = 0
    cum_mdn_rnn_bce = 0
    cum_vae_loss = 0
    cum_vae_rec_loss = 0
    cum_vae_kld_loss = 0

    pbar = tqdm(desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_rep, next_latent_rep = to_latent(obs, next_obs)
        mdn_rnn_prediction = get_mdn_rnn_prediction(
            latent_rep.latent_obs, action)
        get_vae_loss = lambda: vae_loss_function(
            latent_rep.recon_obs, obs,
            latent_rep.obs_mu, latent_rep.obs_logsigma,
            terminal, mdn_rnn_prediction.ds)
        get_mdn_rnn_loss = lambda: mdn_rnn_loss_function(
            latent_rep.latent_obs, action, reward, terminal,
            next_latent_rep.latent_obs, mdn_rnn_prediction)

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
        cum_vae_loss += vae_loss.loss.item()
        cum_vae_rec_loss += vae_loss.reconstruction.item()
        cum_vae_kld_loss += vae_loss.kld.item()

        pbar.set_postfix_str(
            "mdn_loss={mdn_loss:10.6f} bce={bce:10.6f} "
            "gmm={gmm:10.6f} "
            "vae_loss={vae_loss:10.6f} rec={rec:10.6f} "
            "kld={kld:10.6f}".format(
                mdn_loss=cum_mdn_rnn_loss / (i + 1), 
                bce=cum_mdn_rnn_bce / (i + 1),
                gmm=cum_mdn_rnn_gmm / LSIZE / (i + 1),
                vae_loss=cum_vae_loss / (i + 1),
                rec=cum_vae_rec_loss / (i + 1),
                kld=cum_vae_kld_loss / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return (
        cum_mdn_rnn_loss / (i + 1),
        cum_vae_loss / (i + 1))


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
    test(0)

for e in range(epochs):
    if args.test_only:
        break
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
