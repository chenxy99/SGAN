# Every Problem, Every Step, All In Focus: Learning to Solve Vision-Language Problems with Integrated Attention

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

We also provide the [`pretrained model`](https://drive.google.com/file/d/1n7H9Y62uA4cqWrskb0bZuR7IoSyTF1bH/view?usp=share_link), 
and you can directly run the following command to evaluate the performance of the pretrained model on test split.

Reference
------------------
If you find the code useful in your research, please consider citing the paper.
```text
@article{xianyu:2024:sgan,
    Author         = {Xianyu Chen and Jinhui Yang and Shi Chen and Louis Wang and Ming Jiang and Qi Zhao},
    Title          = {Every Problem, Every Step, All In Focus: Learning to Solve Vision-Language Problems with Integrated Attention},
    journal        = {IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI)},
    Year           = {2024}
}
```
