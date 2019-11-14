# CNN Digit Recognition

Simple application for digit recognition with CNN using two different datasets.

 * MNIST Database of Handwritten Digits [link](https://keras.io/datasets/)

 * SVHN - Street View House Numbers Cropped Digit Dataset [link](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_House_Numbers_(SVHN)_Dataset)

## Usage

Project models created in virtual environment using [anaconda](https://www.anaconda.com/).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

```sh
conda env create -f environment.yml
```

Then you can train models with.

```python
python trainer.py
```

Or you can use trained models for predictions.

```python
python predictor.py
```

## Experimental Results

| Trained with Dataset | Test Accuracy on SVHN | Test Accuracy on MNIST |
| :------------------: | :-------------------: | :--------------------: |
| SVHN                 | 0.8877                | 0.5858                 |
| MNIST                | 0.2897                | 0.9905                 |
| SVHN + MNIST         | 0.8983                | 0.9891                 |

### Normalized Confusion Matrices

![svhn-cm-by-svhn](http://furkanomerustaoglu.com/wp-content/uploads/2019/11/svhn_cm_by_svhn.png) ![mnist-cm-by-svhn](http://furkanomerustaoglu.com/wp-content/uploads/2019/11/mnist_cm_by_svhn.png)

![svhn-cm-by-mnist](http://furkanomerustaoglu.com/wp-content/uploads/2019/11/svhn_cm_by_mnist.png) ![mnist-cm-by-mnist](http://furkanomerustaoglu.com/wp-content/uploads/2019/11/mnist_cm_by_mnist.png)

![svhn-cm-by-svhn+mnist](http://furkanomerustaoglu.com/wp-content/uploads/2019/11/svhn_cm_by_svhnmnist.png) ![mnist-cm-by-svhn+mnist](http://furkanomerustaoglu.com/wp-content/uploads/2019/11/mnist_cm_by_svhnmnist.png)
