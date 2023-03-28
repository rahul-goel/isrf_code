import torch
from tqdm import tqdm
import time

@torch.no_grad()
def dev_region_grower(mask_grid, feature_grid, sigma_d, sigma_f, pad_fg):
    torch.cuda.empty_cache()
    DEVICE = "cuda:0"

    # patch borders of full grid
    start_time = time.time()
    padding_array = [1] * 6
    mask_grid_padded = torch.nn.functional.pad(mask_grid, padding_array)
    if pad_fg:
        feature_grid_padded = torch.nn.functional.pad(feature_grid, padding_array)
    feature_grid_padded = feature_grid
    print("\tTime taken to pad the grid.", time.time() - start_time)
    start_time = time.time()
    mask_grid_padded = mask_grid_padded.cuda(torch.device(DEVICE))
    feature_grid_padded = feature_grid_padded.cuda(torch.device(DEVICE))
    print("\tTime taken to move the grids.", time.time() - start_time)

    start_time = time.time()
    mask_grid_padded = mask_grid_padded.squeeze(0).squeeze(0)
    mask_grid = mask_grid.squeeze(0).squeeze(0)
    # mask_grid -> [x, y, z]
    feature_grid_padded = feature_grid_padded.squeeze(0)
    feature_grid = feature_grid.squeeze(0)
    # feature_grid -> [64, x, y, z]

    zero_indices = (mask_grid == 0).nonzero().cuda(torch.device(DEVICE))
    zero_indices += torch.cuda.IntTensor([1, 1, 1], device=torch.device(DEVICE)) # adjust due to padding

    shifts = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                shifts.append([x, y, z])
    shifts = torch.tensor(shifts).cuda(torch.device(DEVICE))

    # do batch wise processing to find out the frontier
    batch_size = 100000
    frontier = []
    for start_idx in range(0, len(zero_indices), batch_size):
        end_idx = start_idx + batch_size
        batch = zero_indices[start_idx:end_idx, :]
        batch_shifted = []
        for shift in shifts:
            batch_shifted.append(batch + shift)
        batch_shifted = torch.stack(batch_shifted, dim=0) # 27, batch_size, 3
        batch_shifted = batch_shifted.reshape(-1, 3) # 27 * batch_size, 3
        values = mask_grid_padded[batch_shifted[:, 0], batch_shifted[:, 1], batch_shifted[:, 2]] # 27 * batch_size, 1
        values = values.reshape(27, -1) # 27, batch_size
        values_sum = values.sum(dim=0)
        frontier.append(batch[values_sum.nonzero().squeeze(-1)])
    frontier = torch.cat(frontier)

    del zero_indices # let's free up some memory

    # do batch wise processing of frontiers
    new_mask_grid_padded = mask_grid_padded.clone().float()
    batch_size = 100000//2

    for start_idx in range(0, len(frontier), batch_size):
        end_idx = start_idx + batch_size
        batch = frontier[start_idx:end_idx, :]

        # pull operation has to be performed on the frontiers
        batch_shifted = []
        for shift in shifts:
            batch_shifted.append(batch + shift)
        batch_shifted = torch.stack(batch_shifted, dim=0) # 27, batch_size, 3
        nbd_mask_values = mask_grid_padded[batch_shifted[..., 0], batch_shifted[..., 1], batch_shifted[..., 2]] # 27, batch_size
        # nbd_feature_values = feature_grid_padded[:, batch_shifted[..., 0], batch_shifted[..., 1], batch_shifted[..., 2]] # 64, 27, batch_size

        nbd_feature_dist = feature_grid_padded[:, batch_shifted[..., 0], batch_shifted[..., 1], batch_shifted[..., 2]] - feature_grid_padded[:, batch[..., 0], batch[..., 1], batch[..., 2]].unsqueeze(1)
        # nbd_feature_dist_tmp = (nbd_feature_dist ** 2).sum(0)
        nbd_feature_dist_tmp = nbd_feature_dist.pow_(2).sum(0)

        del nbd_feature_dist #, nbd_feature_values
        torch.cuda.empty_cache()
        nbd_feature_dist = nbd_feature_dist_tmp
        e_nbd_feature_dist = torch.exp(-nbd_feature_dist / sigma_f)

        nbd_spatial_dist = (batch_shifted - batch.unsqueeze(0)).abs().sum(2)
        e_nbd_spatial_dist = torch.exp(-nbd_spatial_dist / sigma_d)

        sum_weight = (e_nbd_feature_dist * e_nbd_spatial_dist).sum(0)

        batch_new_mask = (nbd_mask_values * e_nbd_feature_dist * e_nbd_spatial_dist).sum(0) / sum_weight
        new_mask_grid_padded[batch[..., 0], batch[..., 1], batch[..., 2]] = batch_new_mask

    new_mask_grid = new_mask_grid_padded[1:-1, 1:-1, 1:-1]
    new_mask_grid = new_mask_grid.unsqueeze(0).unsqueeze(0)

    # unsqueeze the feature grid and mask grid for future use
    mask_grid = mask_grid.unsqueeze(0).unsqueeze(0)
    feature_grid = feature_grid.unsqueeze(0)

    del feature_grid_padded
    torch.cuda.empty_cache()

    new_mask_grid = new_mask_grid
    print("\tTime taken to for the bilateral search.", time.time() - start_time)
    return new_mask_grid

@torch.no_grad()
def dev_region_grower_mask(mask, fg, sigma_d=5.0, sigma_f=0.5, pad_fg=True):
    start_time = time.time()
    mask = dev_region_grower(mask, fg, sigma_d, sigma_f, pad_fg)
    mask[mask > 0.1] = 1.0
    mask[mask <= 0.1] = 0.0
    print("Time taken by one iteration of region growing", time.time() - start_time)
    return mask


if __name__ == "__main__":
    # world_size = [200, 200, 200]
    world_size = [320, 320, 320]

    mask_grid = torch.randint(0, 100, [1, 1, *world_size]) / torch.tensor(99)
    mask_grid = mask_grid.int()
    feature_grid = torch.rand([1, 64, *world_size])

    start_time = time.time()
    dev_region_grower(mask_grid, feature_grid, 5, 5)
    print("Time taken by one iteration of region growing: ", time.time() - start_time)