# Interactive Segmentation of Radiance Fields

[Project Page](https://rahul-goel.github.io/isrf) | [Arxiv](https://arxiv.org/abs/2212.13545) | [Data](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rahul_goel_research_iiit_ac_in/Es4qJ_plQY1Pnqw_OwSzqNQBdtiFrBLUlLi_Da8Fn2Ukvw?e=9AA6M6)

### GUI TOOL For segmentation

Demonstration on the GARDEN scene. This shows the demonstration of positive and negative strokes.

https://user-images.githubusercontent.com/16517445/228188006-3163bc40-6783-44f7-837e-e90aa9497919.mp4

Demonstration on the KITCHEN scene. This shows the demonstration of multiple positive strokes from different views.

https://user-images.githubusercontent.com/16517445/228188085-b6db61e2-da98-4f3e-bc57-716ec4d5d70a.mp4

Please note that there are two branches in this repository. The main branch contains the implementation of bilateral search which uses the spatio-semantic distance. The branch "additional_spaces" is an experimental branch which also accounts for the color latent vector of TensoRF for the calculation of search-distance.

> If you want 2D masks or 3D masks (stored as checkpoints), you can checkout the data we have released. All the masks are created using our method.

> To run the interactive segmentation, we require a GPU with 22-23GB of VRAM, we have tested the code on two consumer level GPUs RTX 3090 and RTX 4090.

> The current configuration has been tested on Ubuntu 22.04. With CUDA 11.8. 

## Dependencies

```
conda create -n isrf python=3.9
conda activate isrf
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
pip install -r requirements.txt
conda install -c conda-forge faiss-gpu
```

[Pytorch](https://pytorch.org) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent, please install the correct version for your machine.

## Data Preparation


#### Datasets - [MipNeRF360](https://jonbarron.info/mipnerf360/), [NeRF-LLFF](https://drive.google.com/drive/folders/14boI-o5hGO9srnWaaogTU5_ji7wkX2S7), [Other LLFF](https://drive.google.com/drive/folders/1M-_Fdn4ajDa0CS8-iqejv0fQQeuonpKF) 
Follow the below directory structure for placing the data:
```
data
├── 360_v2             # Link: MIPNeRF360
│   └── [bicycle|bonsai|counter|garden|kitchen|room|stump]
│       ├── poses_bounds.npy
│       └── [images_2|images_4|images_8]
│
├── nerf_llff_data     # Link: NeRF-LLFF, Other LLFF
│   └── [fern|flower|fortress|horns|leaves|orchids|room|trex|chesstable|...]
│       ├── poses_bounds.npy
│       └── [images_2|images_4|images_8]
```

The code for feature extraction has been taken from [N3F](https://github.com/dichotomies/N3F). Thanks to the original authors for providing it.
Please follow the following instructions to prepare the features:
- To download the DINO checkpoint, run the following command:
    
    Ubuntu
    ```
    cd feature_extractor
    bash download_dino.sh
    ```
- To extract the DINO features and place them in the correct directory, run the following command. Note that we use the images downscaled by a factor of 8.
    ```
    python extract.py --dir_images ../data/nerf_llff_data/horns/images_4
    ```

## Training
To train the radiance field, run the following commands: The first Run is optimize for Radiance Fields, followed by distilling the DINO semantics into the Lattice.
```
cd ..
python run.py --config configs/llff/horns.py --stop_at 20000
python run.py --config configs/llff/horns.py --distill_active --weighted_distill_loss --stop_at 25000
```

## GUI
Once the feature field is trained, the GUI can be launched using:
```
python gui.py --config configs/llff/horns.py
```
If the user wants to change the rendered resolution from given [1,2,4,8], resolution, he can do so by altering the config. file
`configs/llff/horns.py`.

## Citation

If you find our code, data or ideas useful, please cite our work with:

```
@inproceedings{isrfgoel2023,
    title={{Interactive Segmentation of Radiance Fields}}, 
    author={Goel, Rahul and Sirikonda, Dhawal and Saini, Saurabh and Narayanan, P.J.},
    year={2023},
    booktitle = {{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}},
}
```
