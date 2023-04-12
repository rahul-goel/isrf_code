# CHANGELOG

## 04-12-2023

- Users have to train using 64 dimensional feature vectors. But to perform segmentation (which required high memory usage), users can now use the flag `--dino_dim k` to load first `k` DINO features for segmentation (features are in importance order due to PCA). Please note that doing this will lead to less accurate segmentation. The results shown in the paper are using all 64 features.
- Getting HCR is now faster by adding pruning of voxels with low density. 2-3x speedup.
- A sudden GPU memory peak has been removed by moving some calculation to the CPU leading to slightly slower precomputation but overall lower GPU usage at peak.
- Region growing is also 3x faster now. This is also due to pruning of low density points.