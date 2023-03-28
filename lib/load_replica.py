import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import math
from .load_features import load_features


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_replica_data(basedir='./data/replica/office_0', half_res=False, testskip=5, args=None):
    poses = []
    with open(os.path.join(basedir, 'traj_w_c.txt'), 'r') as fp:
        for line in fp:
            tokens = line.split(' ')
            tokens = [float(token) for token in tokens]
            tokens = np.array(tokens).reshape(4, 4)
            poses.append(tokens)
    poses =  np.stack(poses, 0)

    all_imgs_paths = sorted(os.listdir(os.path.join(basedir, 'rgb')))
    all_imgs_paths = all_imgs_paths[::5]
    poses = poses[::5]

    imgs = []


    for i in range(len(all_imgs_paths)):
        fname = os.path.join(basedir, 'rgb', all_imgs_paths[i])
        imgs.append(imageio.imread(fname))
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    hfov = 90
    focal = W / 2.0 / math.tan(math.radians(hfov / 2.0))

    if args is not None and args.distill_active:
        fts_dict = load_features(file=os.path.join(basedir, "features.pt"), imhw=(H, W), selected=all_imgs_paths)
        fts = []
        for i in range(len(all_imgs_paths)):
            fname = os.path.join(basedir, 'rgb', all_imgs_paths[i])
            just_fname = fname.split('/')[-1]
            fts.append(fts_dict[just_fname].permute(1, 2, 0))
        fts = torch.stack(fts)


    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    i_split = [np.arange(0, len(poses)), np.arange(0, len(poses), 4), np.arange(0, len(poses), 4)]

    if args is not None and args.distill_active:
        return imgs, poses, render_poses, [H, W, focal], i_split, fts
    else:
        return imgs, poses, render_poses, [H, W, focal], i_split, None

if __name__ == "__main__":
    load_replica_data()
