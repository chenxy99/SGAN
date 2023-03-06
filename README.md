# Learning to Solve Real-World Problems with Graph-Based Attention

This code implements the prediction of solution graph of for real-world problems by incorporating a novel integrated attention mechanism considering the importance of
features within each step as well as across multiple steps.

Disclaimer
------------------
We adopt the pipeline implementation of GPO [`vse-infty`](https://github.com/woodfrog/vse_infty) and slightly modify it to accommodate our pipeline.
We appreciate to the library PyTorch Geometric [`PyG`](https://pytorch-geometric.readthedocs.io/en/stable/) with providing Graph Neural Networks (GNNs) modules for a wide range of applications related to structured data 


Requirements
------------------

- Python 3.8
- PyTorch 1.12.1 (along with torchvision)

- We also provide the conda environment ``environment.yml``, you can directly run

```bash
$ conda env create -f environment.yml
```

to create the same environment where we successfully run our codes.


Data Preparation
------------------
We use [`VisualHow`](https://github.com/formidify/VisualHow) to train our SGAN models, 
which is dataset for a free-form and open-ended research that focuses on understanding the real-life problem and 
providing the corresponding solutions by incorporating key component across multiple modalities with complex dependencies between steps. 
Please follow the link to download the corresponding data.


Model Training
------------------
If you plan to train the model from scratch, we provide the follow script and you can run it by:
```bash
$ sh bash/train_grid_visualhow.sh
```

Model Testing
------------------
If you complete the model training, you can run the following commands to evaluate the performance of your trained model on test split.
```bash
$ python eval.py --split test
```

Reference
------------------
If you find the code useful in your research, please consider citing the paper.
```text
TBD
```