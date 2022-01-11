# MM-Net: 5th Place Solution to PRCV Challenge 2021 Alzheimer's Disease Classification Task
This repo contains the supported pytorch code and configuration files to reproduce alzheimer's disease classification results of [MM-Net](https://developer.huaweicloud.com/develop/aigallery/algorithm/detail?id=18ab4679-279c-4f41-af64-3e90ec583fdf). 
		Official [website](https://competition.huaweicloud.com/information/1000041489/circumstance?zhishi=) of the competition. Link to our team's [Huawei homepage](https://marketplace.huaweicloud.com/markets/aihub/modelhub/detail/?id=18ab4679-279c-4f41-af64-3e90ec583fdf).

![Overall  Architecture](img/MMNet.png?raw=true)

## Environment
Prepare an environment with python=3.6, and then run the command "pip install -r requirements.txt" for the dependencies.

## Data Preparation
- For experiments we used one dataset:
    - Clinical sMRI: https://competition.huaweicloud.com/information/1000041489/circumstance?zhishi=

- File structure
    ```
      train_data
      |--- train
      |   |--- Subject_xxxx.npy
      |   |--- Subject_xxxx.npy
      |   |--- ...
      |   |--- train_open.csv
      MM-Net
      |---model.py
      |---customize_service.py
      |---std.npy
      |---mean.npy
      |---pip-requirements.txt
      ...
    ```

## Pre-Trained Base Model For PRCV Challenge

- AD-CLS: https://marketplace.huaweicloud.com/markets/aihub/modelhub/detail/?id=18ab4679-279c-4f41-af64-3e90ec583fdf
- Download AD-CLS pre-trained model and add it under MM-Net folder before running test.py

## Train/Test
The entries of this competition are deployed on Huawei Cloud to run and test, and if you want to run locally, you need to modify the inference code.
- Train : Run the train script on PRCV 2021 Training Dataset with Base model Configurations. 
```bash
python model.py --train_url your_path --data_url your_data_path
```

- Test : Run the test script on PRCV 2021 Training Dataset. 
```bash
python customize_service.py 
```

## Acknowledgements
Thanks to Huawei Cloud for providing the competition platform.
