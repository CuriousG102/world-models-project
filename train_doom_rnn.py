""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as f
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
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

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--datasets', type=str, default='datasets',
                    help='Where the datasets are stored')
args = parser.parse_args()

cuda = torch.cuda.is_available()
print('CUDA: {}'.format(cuda))
device = torch.device('cuda' if cuda else 'cpu')

# constants
ASIZE = 1
BSIZE = 16
SEQ_LEN = 70  # 2 seconds
epochs = 30
LSIZE = 64
RSIZE = 512

# Loading VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file, map_location={'cuda:0': str(device)})
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    mkdir(rnn_dir)

mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file, map_location={'cuda:0': str(device)})
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


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
transform = transforms.Lambda(
    lambda x: x / 255)
train_loader = DataLoader(
    VariableLengthRolloutSequenceDataset(join(args.datasets#, 'doom'
        ), SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, collate_fn=collation_fn)
test_loader = DataLoader(
    VariableLengthRolloutSequenceDataset(join(args.datasets#, 'doom'
        ), SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=8, collate_fn=collation_fn)

def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        (obs, lengths), (next_obs, _) = [pad_packed_sequence(o) for o in (obs, next_obs,)]
        max_length, batch_size = obs.shape[:2]
        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            [y.reshape((max_length, batch_size, LSIZE)) 
             for y in vae(x.reshape((-1, 3, SIZE, SIZE)))[1:]]
            for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu))
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]

    return pack_padded_sequence(latent_obs, lengths), pack_padded_sequence(latent_next_obs, lengths)

def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs):
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
    mus, sigmas, logpi, rs, ds = mdrnn(action.unsqueeze(-1), latent_obs)
    gmm_losses = []
    bce_losses = []
    for b, length in enumerate(lengths):
        gmm_losses.append(
            gmm_loss(latent_next_obs[:length, b], mus[:length, b], sigmas[:length, b], logpi[:length, b]))
        bce_losses.append(
            f.binary_cross_entropy_with_logits(ds[:length, b], terminal[:length, b]))
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

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0

    pbar = tqdm(desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "gmm={gmm:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 gmm=cum_gmm / LSIZE / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss / (i + 1)


train = partial(data_pass, train=True)
test = partial(data_pass, train=False)

cur_best = None
for e in range(epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(rnn_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": mdrnn.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    rnn_file)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        break
