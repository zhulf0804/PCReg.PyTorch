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
    # Iterative Benchmark
    python modelnet40_evaluate.py --root your_data_path/modelnet40_ply_hdf5_2048 --checkpoint checkpoint_path/test_min_degree_error.pth --method benchmark --cuda
    
    # Visualization
    # python modelnet40_evaluate.py --root your_data_path/modelnet40_ply_hdf5_2048 --checkpoint checkpoint_path/test_min_degree_error.pth --method benchmark  --show
    
    # ICP
    # python modelnet40_evaluate.py --root your_data_path/modelnet40_ply_hdf5_2048 --method icp
    
    # FGR
    # python modelnet40_evaluate.py --root your_data_path/modelnet40_ply_hdf5_2048 --method fgr --normal

    ```

- train
    
    ```
    CUDA_VISIBLE_DEVICES=0 python modelnet40_train.py --root your_data_path/modelnet40_ply_hdf5_2048
    ```

## Experiments


- Point-to-Point Correspondences

| Method | isotropic R | isotropic t | anisotropic R(mse, mae) | anisotropic t(mse, mae) | time(s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ICP | 11.84 | 0.17 | 18.47(5.86) | 0.23(0.08) | 0.07 |
| FGR | 0.01 | 0.00 | 0.14(0.01) | 0.00(0.00) | 0.19 |
| IBenchmark | 7.90 | 0.10 | 11.17(3.73) | 0.14(0.05) | 0.022 |
| IBenchmark + GN | 6.43 | 0.08 | 10.38(3.02) | 0.13(0.04) | 0.035 |
| IBenchmark + NL | 
| IBenchmark + GN + NL | 

- Partial-to-Complete Registration

**Note**: 
- IBenchmark means `Iterative Benchmark`, GN means `Group Normalization`, NL means `Normal Vectors`.
- Detailed metrics information please refer to [RPM-Net](https://arxiv.org/pdf/2003.13479.pdf)[CVPR 2020].

## Train your Own Data(optimizing, try later..)
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
