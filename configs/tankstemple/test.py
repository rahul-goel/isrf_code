_base_ = '../default.py'

expname = 'dvgo_test'
basedir = './logs/scannet'

data = dict(
    datadir='./data/scannet/scene0000_00',
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
    unbounded_inward=True,
    spherify=True,
    #factor=8,
)

coarse_model_and_render = dict(
maskout_near_cam_vox=False,
)

coarse_train = dict(
N_iters=0,
pervoxel_lr=False,
)

fine_train = dict(
    N_iters=80000,
    N_rand=1024 * 4,
    lrate_decay=80,
    ray_sampler='flatten',
    weight_nearclip=1.0,
    weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000,10000,12000,14000,16000],
    tv_before=20000,
    tv_dense_before=20000,
    weight_tv_density=1e-6,
    weight_tv_k0=1e-7,
)

alpha_init = 1e-4
stepsize = 0.5

fine_model_and_render = dict(
    num_voxels=320**3,
    f_num_voxels=320**3,
    num_voxels_base=160**3,
    f_num_voxels_base=160**3,
    alpha_init=alpha_init,
    stepsize=stepsize,
    fast_color_thres={
        '_delete_': True,
        0   : alpha_init*stepsize/10,
        1500: min(alpha_init, 1e-4)*stepsize/5,
        2500: min(alpha_init, 1e-4)*stepsize/2,
        3500: min(alpha_init, 1e-4)*stepsize/1.5,
        4500: min(alpha_init, 1e-4)*stepsize,
        5500: min(alpha_init, 1e-4),
        6500: 1e-4,
    },
    world_bound_scale=1,
    maskout_near_cam_vox=False,
)
