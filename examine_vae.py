""" Some data examination """
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import torch

from models.vae import VAE
from utils.misc import LSIZE

import argparse
import os
from os.path import exists

# PyTorch setup
cuda = torch.cuda.is_available()
print('CUDA: {}'.format(cuda))
torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if cuda else "cpu")

parser = argparse.ArgumentParser(description='Dataset examination')
parser.add_argument('--datasets', type=str, default='datasets',
                    help='Where the datasets are stored')
parser.add_argument('--vae', type=str, help='VAE checkpoint')
parser.add_argument('--sleep_time', type=float, default=.1)

args = parser.parse_args()

assert exists(args.vae), "No trained VAE in the originallogdir..."
state = torch.load(args.vae, map_location={'cuda:0': str(device)})
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))
vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

def transform(x):
    return torch.Tensor(
        np.expand_dims(
            np.transpose(x, (2, 0, 1)) / 255,
            axis=0))

def plot_rollout():
    """ Plot a rollout """
    from torch.utils.data import DataLoader
    from data.loaders import RolloutSequenceDataset
    dataloader = DataLoader(
        RolloutSequenceDataset(
            root=os.path.join(args.datasets, 'carracing'), seq_len=900,
            transform=lambda x: x, buffer_size=10,
            train=False),
        batch_size=1, shuffle=True)

    dataloader.dataset.load_next_buffer()

    # setting up subplots
    plt.subplot(2, 1, 1)
    monitor_obs = plt.imshow(np.zeros((64, 64, 3)))
    plt.subplot(2, 1, 2)
    monitor_rec_obs = plt.imshow(np.zeros((64, 64, 3)))

    for data in dataloader:
        obs_seq = data[0].numpy().squeeze()
        action_seq = data[1].numpy().squeeze()
        for obs, action in zip(obs_seq, action_seq):
            monitor_obs.set_data(obs.astype(np.uint8))
            with torch.no_grad():
                monitor_rec_obs.set_data(np.transpose(vae(transform(obs))[0].squeeze(), (1, 2, 0)))
            print(action)
            plt.pause(.01)
        break

if __name__ == '__main__':
    plot_rollout()
