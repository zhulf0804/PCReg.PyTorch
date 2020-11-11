## Introduction

A Simple Point Cloud Registration Pipeline based on Deep Learning. Detailed Information Please Visit this [Zhihu Blog](). 

## Install
- requirements.txt `pip install -r requirements.txt`
- open3d-python==0.9.0.0 `python -m pip install open3d==0.9`
- emd loss `cd loss/cuda/emd_torch & python setup.py install`


## Start
- Download data from [[here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip), `435M`]
- evaluate and show(download the pretrained checkpoint from [[here]() `17.9 M`] first)

    ```
    python modelnet40_evaluate.py --root /root/data/modelnet40_ply_hdf5_2048 --checkpoint work_dirs/1106convbn/checkpoints/test_min_loss.pth --method benchmark --cuda
    
    # python modelnet40_evaluate.py --root /root/data/modelnet40_ply_hdf5_2048 --checkpoint work_dirs/1106convbn/checkpoints/test_min_loss.pth --method benchmark --cuda --show #Visualization
    
    ```

- train
    
    ```
    CUDA_VISIBLE_DEVICES=0 python modelnet40_train.py --root /root/data/modelnet40_ply_hdf5_2048
    ```

## Experiments

| Method | mse_t | mse_R | mse_degree | time(s) |
| :---: | :---: | :---: | :---: | :---: |
| Iterative Benchmark | 0.35 | 0.21 | 9.37 | 0.02 |
| icp | 0.40 | 0.38 | 11.86 | 0.06 |


## Acknowledgements

Thanks for the open source [code](https://github.com/vinits5/pcrnet_pytorch) for helping me to train the Point Cloud Registration Network successfully.
