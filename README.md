# AMSR-Few-Shot
Adapting Multi-source Representations for Cross-Domain Few-shot Learning (CD-FSL)

## Enviroment

Python 3.6

Pytorch 1.6

h5py

## Datasets
The following datasets are used for evaluation:

### Source domain: 

* miniImageNet 
* CUB (http://www.vision.caltech.edu/visipedia/CUB-200.html)
* CIFAR100
* Caltech256
* DTD (<https://www.robots.ox.ac.uk/~vgg/data/dtd/>)

### Target domains: 

* **EuroSAT**:

    Home: http://madm.dfki.de/downloads

    Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

* **ISIC2018**:

    Home: http://challenge2018.isic-archive.com

    Direct (must login): https://challenge.isic-archive.com/data#2018

* **Plant Disease**:

    Home: https://www.kaggle.com/saroz014/plant-disease/

    Direct: command line `kaggle datasets download -d plant-disease/data`

* **ChestX-Ray8**:

    Home: https://www.kaggle.com/nih-chest-xrays/data

    Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

## Steps

1. Download the datasets using the above links.

2. Download miniImageNet using <https://drive.google.com/file/d/1uxpnJ3Pmmwl-6779qiVJ5JpWwOGl48xt/view?usp=sharing>

3. Change configuration file `./configs.py` to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.

4. Train multi-representations on miniImageNet, CUB, CIFAR100, DTD, Caltech256

    • *Train multi-representations on 5 datasets from scratch* (optinal)

    ```bash
        python train_multi_representations.py --dataset multi_source --model WResNet12 --method multi_domain --train_aug --stop_epoch 250 --save_freq 50
    ```
    • *Or download the pre-trained model by the following link*  
        <https://sjtueducn-my.sharepoint.com/:u:/g/personal/liu_ge_sjtu_edu_cn/EeJxNcfvK45Ik1mogi8uLlAB8WCYrNZSyc36825r65JPGw?e=d6NxN9>  
        Put the model file to the path "./logs/checkpoints/multi_source/WResNet12_multi_domain_aug/"
5. Test on the target dataset
    • *For example*

    ```bash
        python test_multi_domain_pesudo_label.py --dataset multi_source --model WResNet12 --method multi_domain --train_aug --n_shot 5 --test_dataset ISIC
    ```