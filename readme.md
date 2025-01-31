# ArchBERT: Bi-Modal Understanding of Neural Architectures and Natural Languages
This repo contains the implementation and demo of the paper ['ArchBERT: Bi-Modal Understanding of Neural Architectures and Natural Languages'](https://aclanthology.org/2023.conll-1.7/) that has been accepted to appear at the CoNLL2023 conference.

Link to arXiv paper: https://arxiv.org/abs/2310.17737

![](https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/framework.png)

### Download Code and Datasets
```
!wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/code.zip
!unzip -qo code.zip
```

### Package Requirements
- Anaconda (version 2020.07)
- All the other requirements are listed in **environment.yml** file
- After installing Anaconda, use the following command to create an conda environment with the required packages:


```python
!conda env create -f environment.yml
!conda activate archbert
```

### AutoNet dataset generation
The code for creating AutoNet and AutoNet-AQA train/val sets (with e.g., 100 neural architectures):


```python
!python ./data/autonet_generator.py train 100 ./data/datasets/autonet default default
!python ./data/autonet_generator.py val 100 ./data/datasets/autonet default default
!python ./data/autonet_generator.py train 100 ./data/datasets/autonet_qa qa multi
!python ./data/autonet_generator.py val 100 ./data/datasets/autonet_qa qa multi
```

### TVHF dataset generation
Run the following command to generate the TVHF train and validation sets.
- path: the path to save the generated dataset
- num_nets: the number of architectures to be generated


```python
!python ./data/tvhf_dataset_generator --path=./datasets/tvhf/ --num_nets=5
```

### Architecture Reasoning (AR) on TVHF
- num_nets: the number of architectures to be evaluated


```python
!python test_archbert.py
        --task=reasoning
        --dataset=tvhf
        --batch_size=1
        --layernorm
        --cross_encoder
        --data_dir=./data/datasets/tvhf
        --model_dir=./pretrained-models/archbert_tvhf
        --validate
        --num_nets=100
```

### Architecture Clone Detection (ACD) on TVHF


```python
!python test_archbert.py
        --task=na_clone_detection
        --dataset=tvhf
        --batch_size=1
        --layernorm
        --cross_encoder
        --data_dir=./data/datasets/tvhf
        --model_dir=./pretrained-models/archbert_tvhf
        --validate
        --num_nets=100
```

### Architecture Captioning (AC) on AutoNET


```python
!python test_archbert.py
        --task=langdec
        --dataset=autonet
        --batch_size=1
        --layernorm
        --cross_encoder
        --data_dir=./data/datasets/autonet
        --model_dir=./pretrained-models/archbert_autonet_ac
        --validate
        --num_nets=100
```

### Video demo for different downstream tasks

#### Architectural Reasoning (AR)


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AR_TVHF.mp4" type="video/mp4">
</video>
```


<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AR_TVHF.mp4" type="video/mp4">
</video>



#### Architecture Search (AS)


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AS_TVHF.mp4" type="video/mp4">
</video>
```


<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AS_TVHF.mp4" type="video/mp4">
</video>



#### Architecture Clone Detection (ACD)


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/ACD_TVHF.mp4" type="video/mp4">
</video>
```


<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/ACD_TVHF.mp4" type="video/mp4">
</video>



#### Bi-Modal Architecture Clone Detection (BACD)


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/BACD_TVHF.mp4" type="video/mp4">
</video>
```


<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/BACD_TVHF.mp4" type="video/mp4">
</video>



#### Architecture Captioning (AC)


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AC_AutoNet.mp4" type="video/mp4">
</video>
```


<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AC_AutoNet.mp4" type="video/mp4">
</video>



#### Architectural Question Answering (AQA)


```python
%%HTML
<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AQA_AutoNet.mp4" type="video/mp4">
</video>
```


<video width="1280" controls>
    <source src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/archbert/demo/AQA_AutoNet.mp4" type="video/mp4">
</video>


