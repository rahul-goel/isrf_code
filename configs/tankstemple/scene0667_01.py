_base_ = '../default.py'

expname = 'dvgo_test'
basedir = './logs/scannet'

data = dict(
    datadir='./data/scannet/scene0667_01',
    dataset_type='tankstemple',
    inverse_y=True,
    load2gpu_on_the_fly=True,
    white_bkgd=True,
    unbounded_inward=True,
)

coarse_train = dict(N_iters=0)

#coarse_model_and_render = dict(
#    num_voxels=1024000,           # expected number of voxel
#    num_voxels_base=1024000,      # to rescale delta distance
#    f_num_voxels=1024000,           # expected number of voxel
#    f_num_voxels_base=1024000,      # to rescale delta distance
#    density_type='DenseGrid',     # DenseGrid, TensoRFGrid
#    k0_type='TensoRFGrid',        # DenseGrid, TensoRFGrid
#    f_k0_type='TensoRFGrid',        # DenseGrid, TensoRFGrid
#    density_config=dict(),
#    k0_config=dict(n_comp=48),
#    f_k0_config=dict(n_comp=64),
#    mpi_depth=128,                # the number of planes in Multiplane Image (work when ndc=True)
#    nearest=False,                # nearest interpolation
#    pre_act_density=False,        # pre-activated trilinear interpolation
#    in_act_density=False,         # in-activated trilinear interpolation
#    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
#    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
#    rgbnet_dim=0,                 # feature voxel grid dim
#    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
#    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
#    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
#    rgbnet_width=128,             # width of the colors MLP
#    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
#    fast_color_thres=1e-7,        # threshold of alpha value to skip the fine stage sampled point
#    maskout_near_cam_vox=False,    # maskout grid points that between cameras and their near planes
#    world_bound_scale=1,          # rescale the BBox enclosing the scene
#    stepsize=0.5,                 # sampling stepsize in volume rendering
#)

fine_model_and_render = dict(
    maskout_near_cam_vox=False,
)

#coarse_train = dict(
#    pervoxel_lr=False,
#    #pervoxel_lr_downrate=2,
#)
