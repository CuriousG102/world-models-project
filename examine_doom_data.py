""" Some data examination """
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import argparse
import os

parser = argparse.ArgumentParser(description='Dataset examination')
parser.add_argument('--datasets', type=str, default='datasets',
                    help='Where the datasets are stored')
parser.add_argument('--example_num', type=int)

args = parser.parse_args()


def plot_rollout():
    """ Plot a rollout """
    from torch.utils.data import DataLoader
    from data.loaders import VariableLengthRolloutSequenceDataset
    dataloader = DataLoader(
        VariableLengthRolloutSequenceDataset(
            root=args.datasets, transform=lambda x: x, 
            buffer_size=10, seq_len=500, train=False),
        batch_size=1)

    # setting up subplots
    plt.subplot(2, 2, 1)
    monitor_obs = plt.imshow(np.zeros((64, 64, 3)))
    plt.subplot(2, 2, 2)
    monitor_next_obs = plt.imshow(np.zeros((64, 64, 3)))
    plt.subplot(2, 2, 3)
    monitor_diff = plt.imshow(np.zeros((64, 64, 3)))

    for i, data in enumerate(dataloader):
        if i != args.example_num:
            continue
        obs_seq = data[0].numpy().squeeze()
        action_seq = data[1].numpy().squeeze()
        next_obs_seq = data[-1].numpy().squeeze()
        for obs, action, next_obs in zip(obs_seq, action_seq, next_obs_seq):
            monitor_obs.set_data(obs.astype(np.uint8))
            monitor_next_obs.set_data(next_obs.astype(np.uint8))
            monitor_diff.set_data(next_obs - obs)
            plt.pause(.01)
        break

if __name__ == '__main__':
    plot_rollout()
