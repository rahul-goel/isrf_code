import torch
import faiss
from time import time
from lib.grid import TensoRFGrid

def hybrid_nnfm(model, thresh, fv, sigma_d = 5.0, sigma_f = 1.0):
    with torch.no_grad():
        print("Starting density filtering using nnfm.")
        model = model.cpu()
        fv = fv.cpu()

        fv = fv.reshape(64, -1)
        fv = torch.unique(fv, dim=1)
        fv = fv.permute(1, 0)

        print("Reconstructing the Feature Grid.")
        start_time = time()
        fg = model.f_k0.get_dense_grid().cpu()
        dg = model.density.grid.cpu()
        fg = fg.permute([0, 2, 3, 4, 1])
        print("Feature Grid ready.", time() - start_time)

        print("Taking nearest neighbor distance.")
        start_time = time()
        dist = torch.cdist(fg, fv)
        #dist = scipy.spatial.distance.cdist(fg.numpy().reshape(-1, 64), fv.numpy().reshape(-1, 64))
        dist = torch.min(dist, -1)[0]
        dist = dist.reshape(dg.shape)
        print("Nearest neighbor distance taken.", time() - start_time)

        mask = (dist < thresh).float()
        fg = fg.permute([0, 4, 1, 2, 3])
        print("Finished density filtering using nnfm.")
        return mask, fg


@torch.no_grad()
def kmeans(fv):
    print("Buliding K Means Model.")
    start_time = time()

    dim = fv.shape[0]
    fv = fv.cpu()
    fv = fv.reshape(dim, -1)
    fv = fv.permute(1, 0)
    kmeans = faiss.Kmeans(d=dim, k=11, niter=100, nredo=1, gpu=True)
    kmeans.train(fv.contiguous())

    print("K Means Model Ready.", time() - start_time)
    return kmeans

@torch.no_grad()
def query_kmeans(kmeans, fg, valid_idx, thresh, xyz):
    print("Querying the feature grid points.")
    start_time = time()

    dist, _ = kmeans.index.search(fg, 1)
    dist = torch.tensor(dist)
    print("Predicted the feature grid points.", time() - start_time)
    print("Creating mask using the queried distance.")
    start_time = time()
    valid_mask = (dist < thresh).float()
    mask = torch.zeros([1, 1, *xyz])
    mask[:, :, valid_idx] = valid_mask.squeeze(-1)

    print("Created mask using the queried distance.", time() - start_time)
    return mask


def hybrid_kmeans(model, thresh, fv):
    with torch.no_grad():
        model = model.cpu()
        fv = fv.cpu()
        fv = fv.reshape(64, -1)
        fv = torch.unique(fv, dim=1)
        fv = fv.permute(1, 0)

        print("Building K Means Model.")
        start_time = time()
        kmeans = faiss.Kmeans(d=64, k=11, niter=300, nredo=10, gpu=1)
        kmeans.train(fv)
        print("K Means Model Ready.", time() - start_time)

        print("Reconstructing the Feature Grid.")
        start_time = time()
        if isinstance(model.f_k0, TensoRFGrid):
            # model.f_k0 = model.f_k0.cuda()
            # fg = get_dense_grid_batch_processing(model.f_k0).cpu().contiguous()
            # model.f_k0 = model.f_k0.cpu()
            fg = model.f_k0.get_dense_grid().cpu() # 1, 64, x, y, z
        else:
            fg = model.f_k0.grid
        xyz = fg.shape[2:]
        fg = fg.squeeze(0).permute(1, 2, 3, 0) # x, y, z, 64
        fg = fg.reshape(-1, 64)
        print("Feature Grid ready.", time() - start_time)

        print("Quering the feature grid points.")
        start_time = time()
        dist, _ = kmeans.index.search(fg, 1)
        dist = torch.tensor(dist)
        print("Predicted the feature grid points.", time() - start_time)


        dg = model.density.grid.cpu()
        dist = dist.reshape(dg.shape)

        mask = (dist < thresh).float()
        fg = fg.reshape(1, *xyz, 64)
        fg = fg.permute([0, 4, 1, 2, 3])
        return mask, fg


def hybrid_average(model, thresh, fv, sigma_d = 5.0, sigma_f = 1.0):
    with torch.no_grad():
        print("Starting density filtering using nnfm.")
        model = model.cpu()
        fv = fv.cpu()

        fg = model.f_k0.get_dense_grid().cpu()
        dg = model.density.grid.cpu()

        fv = fv.reshape(64, -1)
        fv = fv.mean(dim=1)
        fv = fv.reshape([1, 64, 1, 1, 1])

        dist = (((fg - fv) ** 2).sum(dim=1)).sqrt()
        dist = dist.reshape(dg.shape)

        mask = (dist < thresh).float()
        print("Finished density filtering using nnfm.")
        return mask, fg