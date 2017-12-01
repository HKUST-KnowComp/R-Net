# R-Net
  * Tensorflow implementation of [R-NET: MACHINE READING COMPREHENSION WITH
SELF-MATCHING NETWORKS](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). This project is specially designed for the [SQuAD](https://arxiv.org/pdf/1606.05250.pdf) dataset.
  * Should you have any question, please contact [Wenxuan Zhou](wzhouad@connect.ust.hk).

## Requirements
#### General
  * Python >= 3.5
  * unzip, wget
#### Python Packages
  * Tensorflow >= 1.2.0
  * spaCy
  * tqdm

## Usage

To download and preprocess the data, use

```bash
sh download.sh
python config.py --mode prepro
```

Hyper parameters are stored in config.py. To debug/train/test the model, use

```bash
python config.py --mode debug/train/test
```

The default directory for tensorboard log file is `log/event`


## Detailed Implementaion

  * The original paper uses additive attention, which consumes lots of memory. This project adopts scaled dot attention presented in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  * This project adopts sequence-level dropout and embedding dropout presented in [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf).
  * To solve the degradation problem in stacked RNN, outputs of each layer are concatenated to produce the final output.
  * When the loss on dev set increases in a certain period, the learning rate is halved.
  * During prediction, the project adopts search method and bidirectional answer pointer presented in [Machine Comprehension Using Match-LSTM and Answer Pointer](https://arxiv.org/pdf/1608.07905.pdf).

## Current Results

||EM|F1|
|---|---|---|
|original paper|71.1|79.5|
|this project|69.37|78.42|

<img src="img/em.jpg" width="300">

<img src="img/f1.jpg" width="300">