Semantic Segmentation of Spreader and Cell Guide from LiDAR Point Clouds

This repository contains a custom dataset and configuration for training semantic segmentation models on the AQC dataset using the Pointcept platform.

Dataset
-------
The dataset follows the KITTI format and is located in:
pointcept/datasets/aqc.py

To link the dataset to the expected path in Pointcept, use the following command:
ln -s /hy-tmp/datasets/kitti /root/Pointcept/data/aqc

Configuration
-------------
Training configuration files are stored in:
configs/aqc

Training
--------
To train a model, use the following command as an example:
python tools/train.py --config-file configs/aqc/semseg-pt-v2m2-0-base.py --options save_path=exp/aqc/semseg-pt-v2m2-0-base

Note: The train.py script can be modified in pointcept/engines if needed.

Testing
-------
To test a model, use the following command as an example:
python tools/test.py --config-file configs/aqc/semseg-pt-v3m1-4-train.py --options save_path=exp/aqc/semseg-pt-v3m1-5-test weight=exp/aqc/semseg-pt-v3m1-4-train/model/model_best.pth

Note: The test.py script can also be modified in pointcept/engines.

Experiment Results
------------------
Training and test results are saved in:
exp/aqc

This setup allows easy reproduction of experiments using the provided dataset, configurations, and scripts.
