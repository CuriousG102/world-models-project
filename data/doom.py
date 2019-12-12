"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import cv2
import gym
import numpy as np
import random
from torchvision import transforms
from envs import doom as doom_env
from utils.misc import sample_continuous_policy

MAX_FRAMES = 2100
MIN_LENGTH = 100
ACTIONS = [
    np.array(a) for a in 
    [[False, False], [True, False], [False, True]]]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(120),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def generate_data(rollouts, data_dir): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    env = doom_env.ViZDoomWrapper('DoomTakeCover')

    i = -1
    while i < rollouts:
        env.reset()

        a_rollout = []
        s_rollout = []
        r_rollout = []
        d_rollout = []

        action = random.randint(0, len(ACTIONS)-1)
        repeat = random.randint(1, 11)
        for t in range(MAX_FRAMES):
            if t % repeat == 0:
                action = random.randint(0, len(ACTIONS)-1)
                repeat = random.randint(1, 11)
            s, r, done, _ = env.step(ACTIONS[action])
            # print('before', s.shape, s.max(), s.min(), (s==0).sum())
            s = (transform(s.transpose(1,2,0)) * 255.).data.numpy().astype(np.uint8)
            # print('after', s.shape, s.max(), s.min(), (s==0).sum())
            s_rollout.append(s)
            a_rollout.append(action)
            r_rollout.append(r)
            d_rollout.append(done)
            if done or len(s_rollout) == MAX_FRAMES:
                if len(s_rollout) > MIN_LENGTH:
                    i += 1
                    print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                    np.savez(join(data_dir, 'rollout_{}'.format(i)),
                             observations=np.array(s_rollout),
                             rewards=np.array(r_rollout),
                             actions=np.array(a_rollout),
                             terminals=np.array(d_rollout))
                else:
                    print('Failed rollout: {} frames'.format(len(s_rollout)))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir)
