""" Some data examination """
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_packed_sequence

from models.vae import VAE

import argparse
import os
from os.path import exists

LSIZE = 64

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
parser.add_argument('--vae_two', type=str, help='VAE 2 checkpoint')
parser.add_argument('--example_num', type=int)

args = parser.parse_args()

assert exists(args.vae), "No trained VAE in the originallogdir..."
state = torch.load(args.vae, map_location={'cuda:0': str(device)})
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))
vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

assert exists(args.vae_two), "No trained VAE in the originallogdir..."
state = torch.load(args.vae_two, map_location={'cuda:0': str(device)})
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))
vae_two = VAE(3, LSIZE).to(device)
vae_two.load_state_dict(state['state_dict'])

def transform(x):
    return torch.Tensor(
        np.expand_dims(
            np.transpose(x, (2, 0, 1)) / 255,
            axis=0))

def plot_rollout():
    """ Plot a rollout """
    from torch.utils.data import DataLoader
    from data.loaders import VariableLengthRolloutSequenceDataset
    dataloader = DataLoader(
        VariableLengthRolloutSequenceDataset(
            root=os.path.join(
                args.datasets#, 'doom'
                ), seq_len=900,
            transform=lambda x: x, buffer_size=10,
            train=False),
        batch_size=1)

    # setting up subplots
    plt.subplot(1, 3, 1)
    monitor_obs = plt.imshow(np.zeros((64, 64, 3)))
    plt.subplot(1, 3, 2)
    monitor_rec_obs = plt.imshow(np.zeros((64, 64, 3)))
    plt.subplot(1, 3, 3)
    monitor_rec_obs_two = plt.imshow(np.zeros((64, 64, 3)))

    for i, data in enumerate(dataloader):
        if i != args.example_num:
            continue
        obs_seq = data[0].numpy().squeeze()
        action_seq = data[1].numpy().squeeze()
        for j, (obs, action) in enumerate(zip(obs_seq, action_seq)):
            monitor_obs.set_data(obs.astype(np.uint8))
            with torch.no_grad():
                monitor_rec_obs.set_data(np.transpose(vae(transform(obs))[0].squeeze(), (1, 2, 0)))
                monitor_rec_obs_two.set_data(np.transpose(vae_two(transform(obs))[0].squeeze(), (1, 2, 0)))
            print(action)
            # print(j)
            plt.pause(.01)
        break

if __name__ == '__main__':
    plot_rollout()
