## Introduction

A Simple Point Cloud Registration Pipeline based on Deep Learning. Detailed Information Please Visit this [Zhihu Blog](https://zhuanlan.zhihu.com/p/289620126). 

## Install
- requirements.txt `pip install -r requirements.txt`
- open3d-python==0.9.0.0 `python -m pip install open3d==0.9`
- emd loss `cd loss/cuda/emd_torch & python setup.py install`


## Start
- Download data from [[here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), `435M`]
- evaluate and show(download the pretrained checkpoint from [[Baidu Disk](https://pan.baidu.com/s/1DlGBDR0RLdJ1qxUYqSszdw) `17.15 M`] with password **`0pfg`** first)

    ```
    python modelnet40_evaluate.py --root your_data_path/modelnet40_ply_hdf5_2048 --checkpoint checkpoint_path/test_min_degree_error.pth --method benchmark --cuda
    
    # ICP
    # python modelnet40_evaluate.py --root your_data_path/modelnet40_ply_hdf5_2048 --method icp

    # Visualization
    # python modelnet40_evaluate.py --root your_data_path/modelnet40_ply_hdf5_2048 --checkpoint checkpoint_path/test_min_degree_error.pth --method benchmark  --show
    
    ```

- train
    
    ```
    CUDA_VISIBLE_DEVICES=0 python modelnet40_train.py --root your_data_path/modelnet40_ply_hdf5_2048
    ```

## Experiments

| Method | mse_t | mse_R | mse_degree | time(s) |
| :---: | :---: | :---: | :---: | :---: |
| icp | 0.40 | 0.38 | 11.86 | 0.06 |
| Iterative Benchmark | **0.35** | **0.18** | **7.90** | **0.02** |


| Method | mse_R(mae_R) | mse_t(mae_t) | mse_degree | abc | MCD | time(s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| icp | 0.40 | 0.38 | 11.86 | 0.06 |
| Iterative Benchmark | **0.35** | **0.18** | **7.90** | **0.02** |

MCD means modified CD Distance, detailed information please refer to [RPM-Net](https://arxiv.org/pdf/2003.13479.pdf).

## Train your Own Data
- Prepare the data in the following structure
    ```
    |- CustomData(dir)
        |- train_data(dir)
            - train1.pcd
            - train2.pcd
            - ...
        |- val_data(dir)
            - val1.pcd
            - val2.pcd
            - ...
    ```
- Train
    ```
    python custom_train.py --root your_datapath/CustomData --train_npts 2048 
    # Note: train_npts depends on your dataset
    ```
- Evaluate
    ```
    # Evaluate, infer_npts depends on your dataset
    python custom_evaluate.py --root your_datapath/CustomData --infer_npts 2048 --checkpoint work_dirs/models/checkpoints/test_min_degree_error.pth --method benchmark --cuda
    
    # Visualize, infer_npts depends on your dataset
    python custom_evaluate.py --root your_datapath/CustomData --infer_npts 2048 --checkpoint work_dirs/models/checkpoints/test_min_degree_error.pth --method benchmark --show
    ```

## Acknowledgements

Thanks for the open source [code](https://github.com/vinits5/pcrnet_pytorch) for helping me to train the Point Cloud Registration Network successfully.
