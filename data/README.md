# ArchBERT

### AutoNet dataset generation

The code for creating AutoNet train set (with e.g., 100 neural architectures):
```
python autonet_generator.py train 100 ./datasets/autonet default default
```
The code for creating AutoNet val set (with e.g., 100 neural architectures):
```
python autonet_generator.py val 100 ./datasets/autonet default default
```
The code for creating AutoNet-AQA train set (with e.g., 100 neural architectures):
```
python autonet_generator.py train 100 ./datasets/autonet_qa qa multi
```
The code for creating AutoNet-AQA val set (with e.g., 100 neural architectures):
```
python autonet_generator.py val 100 ./datasets/autonet_qa qa multi
```

### TVHF dataset generation
Run the following command to generate the TVHF train and validation sets:
```
python tvhf_dataset_generator --path=./datasets/tvhf/ --num_nets=5
```
- path: the path to save the generated dataset
- num_nets: the number of architectures to be generated
