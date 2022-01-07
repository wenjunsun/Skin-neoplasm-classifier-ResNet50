# HAM10000 classification with ResNet50 (PyTorch)
## The dataset is __[Skin Cancer MNIST: HAM10000 from Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)__.
## steps to run this program on HYAK:
1. `python prepare_data.py`
2. `python train.py`
3. run `evaluate.ipynb` to evaluate the model. (see `evaluate.ipynb` for results from the experiment)
# Tensorboard visualization of training process
visit [Tensorboard](https://tensorboard.dev/experiment/fBMZZyWbRVSJlfbqQuyZnA/#scalars) for a visualization of training and validation loss and so on throughout the experiment. As we can see, the validation loss and accuracy have a lot of fluctuations.
