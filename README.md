# Bias-Gradient-Regression
Improve gradient accuracy of neural network with appended bias layer

Gradient error (in this case, force error) is harder to reduce comparing to prediction error (energy error). Thus, the idea of this work is to put most of the computing power on force training and fine-tune the model with less epochs on energy training. Utilizing the fact that training on bias layers do not affect gradient of the model, this work propose training on force only first for majority of the epochs follow by energy only training on an appended bias layer for several more epochs. 

Benchmarked models: 
- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)] 

## Requirements
To install requirements with conda:
```setup
conda env create -f environment.yml
```

## Data
- Download the dataset for experimeants manually from [MD17](http://www.sgdml.org/#datasets) or run:
```download
cd ./datasets/MD17 
sh download_datas.sh 
```
A default random split of 80% train, 10% valid and 10% test will be generated.  

## Training
- To train the model(s), run below:
```train
cd ./scirpts
python train.py --root <path_to_data> -M <model_name> -m <molecule_abbreviation> -A <activation_function> --lr2 <lr_last_layer>
```

## Evaluation
In our training script, the evalueation would be conducted and recorded to Wandb right after training is finished.  
- To manually evaluate a trained chemistry model, run:
```eval
cd ./scirpts
python eval.py --root <path_to_data> --model_path <path_to_model_dir> --epoch <select_trained_epoch>
```

## Pre-trained Models
You can download pretrained models here:
[Schnet pretrain](https://drive.google.com/drive/folders/1PXVkEkVWZP1oDGIpjUtK2gvTSpJgB55x?usp=sharing) trained on MD17. 
Other configs can be found in info.txt in each checkpoint folder.

## Results
Our model achieves the following performance on MD17 with metrics Energy MAE, Force MAE, EFwT defined by [Open Catalyst 2020 (OC20) Dataset and Community Challenges](https://opencatalystproject.org/leaderboard.html).

| Model name                   | Energy MAE           | Force MAE          | EFwT           |
| ---------------------------- |--------------------- | ------------------ | -------------- |
| Schnet-bias (Aspirin)        | 242.458 -> **0.149** | 0.442 -> **0.262** | 0 -> **0.071** |
| Schnet-bias (Benzene)        | 14.520 -> **0.075**  | 0.276 -> **0.192** | 0 -> **0.513** |
| Schnet-bias (Ethanol)        | 46.123 -> **0.056**  | 0.253 -> **0.128** | 0 -> **0.904** |
| Schnet-bias (Malonaldehyde)  | 190.557 -> **0.087** | 0.378 -> **0.209** | 0 -> **0.637** |
| Schnet-bias (Naphthalene)    | 101.642 -> **0.134** | 0.256 -> **0.150** | 0 -> **0.752** |
| Schnet-bias (Salicylic acid) | 197.106 -> **0.115** | 0.314 -> **0.262** | 0 -> **0.060** |
| Schnet-bias (Toluene)        | 118.788 -> **0.106** | 0.304 -> **0.173** | 0 -> **0.492** |
| Schnet-bias (Uracil)         | 440.164 -> **0.127** | 0.376 -> **0.265** | 0 -> **0.030** |

Each cell show the improvement of this work from original structure: original_model_error -> biased_model_error, better one in bold.  
Above results are also presented in ./experiments/Schnet/Schnet.ipynb and models to produce this result are available in Pre-train Models section.

## Contributing
Copyright (C) 2023 Kai Chieh Lo

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

