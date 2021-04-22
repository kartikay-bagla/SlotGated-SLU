# Slot-Gated Modeling Using PyTorch

In [Slot-Gated Modeling for Joint Slot Filling and Intent Prediction by Goo et al., 2018](https://www.csie.ntu.edu.tw/~yvchen/doc/NAACL18_SlotGated.pdf), the source code is only given in tensorflow. This repository aims to recreate the model using PyTorch.

## Requirements
* python 3.9
* torch==1.8.1
* tensorboard==2.5.0
* for CPU: `pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
* for CUDA 11.1: `pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
* for CUDA 10.2: `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`


## Usage
`python train.py --dataset=atis` or `python test.py --dataset=atis` to train the model or to run the model on the dataset's validation set respectively.


### References
* [Bi-LSTM Pytorch Question on Stack Overflow](https://stackoverflow.com/questions/53010465/bidirectional-lstm-output-question-in-pytorch)
* [TF v1 Embeddings Documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/embedding_lookup)
* [TF Conv2D Documentation](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/conv2d)
* [TF v1 Bi Directional RNN Documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/bidirectional_dynamic_rnn)
* [TF v1 Basic LSTM Cell Documentation](https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/BasicLSTMCell)
* [Pytorch Conv2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
* [Pytorch Embeddings](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
* [Pytorch LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)
* [Pytorch CrossEntropyLoss](https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
* [Pytorch "SAME" padding equivalent of keras](https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/2)
* [TF rnn_cell_impl._linear](https://thetopsites.net/article/52764340.shtml)
* [Tensorboard with Pytorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)