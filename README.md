# SIDP

The PyTorch implementation of 'Robust Differentially Private Training of Deep Neural Networks'. 
https://arxiv.org/abs/2006.10919

## Usage

To run the code on `MNIST` and `CIFAR-10` datasets, execute `vision.py` with desired parameters:

```
python vision.py 
```

For the text classification model based on the AGNews Corpus:
```
python main_text_class.py [experiment_name] [noise_std] [clip]
python main_text_class.py sidp 0.3 7
```


## Results

#### MNIST

Privacy epsilon |  ∞  |  7  |  3  |  1  | 0.5  | 0.1  | 0.05 | 0.025 |
----------------|-----|-----|-----|-----|------|------|------|-------|
DPSGD (LeNet5)  | 99.2 | 97 | 96.34| 94.11 | 91.1 | 83.0 | 78.96 | 31.56 |
SI-DPSGD (LeNet5) | 99.2 | 98.9 |  98.9 | 98.72 | 99.1 | 99.0 | 98.84 | 90.82 |
SI-DPSGD (BN-LeNet5) | 99.2| 99.17| 99.17| 99.15| 99.18| 99.14| 99.12| 98.58|

#### CIFAR-10

Privacy epsilon |  ∞  |  8  | 4   |  2  |   1  | 0.5  | 0.1  |  0.05 |
----------------|-----|-----|-----|-----|------|------|------|-------|
DPSGD (TF-tutorial)| 80.0 | 73.0 |70.0 | 67.0 | NA |NA |NA |NA|
SI-DPSGD (TF-tutorial)| 80.0| 78.10 | 77.70 |76.0 | 76.05 | 74.20 | 73.80 | 74.05|
SI-DPSGD (ResNet-18)| 93.50 | 90.20 | 90.16 | 90.26 | 90.09 | 89.67 | 84.88 | 84.47 |

#### AGNews Text Classification

Privacy epsilon |  ∞  |  7  | 3   |  1  |   0.5  | 0.1  | 0.05  |
----------------|-----|-----|-----|-----|------|------|------|
DPSGD (BiLSTM-DL)| 88.5 | 83.9 |80.0 | 81.1 | 77.9 | 37.5 |31.8 |
SI-DPSGD (BiLSTM-DL)| 88.5| 85.9 | 85.7 | 83.3 | 81.2 | 77.9 | 56.7 |
DPSGD (LN-BiLSTM-DL)| 88.5 | 83.5 | 82.4 | 82.0 | 78.9 | 50.1 | 31.6 |
SI-DPSGD (LN-BiLSTM-DL)| 88.5 | 87.8 | 87.6 | 85.7 | 85.4 | 84.3 | 80.1 |

#### References 

* PyTorch implementation of DPSGD is taken from https://github.com/ebagdasa/pytorch-privacy. 






