# CNN Digit Recognition

Simple application for digit recognition with CNN using four different datasets.

 * ARDIS-IV - The Swedish Dataset of Historical Handwritten Digits [link](https://ardisdataset.github.io/ARDIS/)

 * MNIST Database of Handwritten Digits [link](https://keras.io/datasets/)

 * ORHD - Optical Recognition of Handwritten Digits Data Set [link](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)

 * SVHN - Street View House Numbers Cropped Digit Dataset [link](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_House_Numbers_%28SVHN%29_Dataset)

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

Environment with tensorflow 2:

```sh
conda env create -f environment.yml
```

Environment with tensorflow 2 without GPU support:

```sh
conda env create -f environment-without-gpu.yml
```

You can use the version you want for training as follows. Valid versions => ["v1", "v2"], default version is "v2"

```python
python trainer.py {version}
```

## Experimental Results

### Using v1 model

| Trained with Dataset    | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS                   | 0.98                   | 0.54                   | 0.63                  | 0.15                  |
| MNIST                   | 0.64                   | 0.99                   | 0.72                  | 0.28                  |
| ORHD                    | 0.31                   | 0.25                   | 0.97                  | 0.13                  |
| SVHN                    | 0.25                   | 0.59                   | 0.45                  | 0.90                  |

| Trained with 2 Dataset  | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS+MNIST             | 0.97                   | 0.99                   | 0.73                  | 0.19                  |
| ARDIS+ORHD              | 0.98                   | 0.56                   | 0.99                  | 0.17                  |
| ARDIS+SVHN              | 0.95                   | 0.76                   | 0.52                  | 0.90                  |
| MNIST+ORHD              | 0.69                   | 0.99                   | 0.98                  | 0.28                  |
| MNIST+SVHN              | 0.60                   | 0.99                   | 0.64                  | 0.90                  |
| ORHD+SVHN               | 0.31                   | 0.60                   | 0.97                  | 0.89                  |

| Trained with 3 Dataset  | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS+MNIST+ORHD        | 0.98                   | 0.99                   | 0.98                  | 0.22                  |
| ARDIS+MNIST+SVHN        | 0.96                   | 0.99                   | 0.69                  | 0.88                  |
| ARDIS+ORHD+SVHN         | 0.97                   | 0.77                   | 0.96                  | 0.86                  |
| MNIST+ORHD+SVHN         | 0.63                   | 0.99                   | 0.98                  | 0.90                  |

| Trained with 4 Dataset  | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS+MNIST+ORHD+SVHN   | 0.97                   | 0.99                   | 0.98                  | 0.90                  |

### Using v2 model

Used number of filters/kernels increased

| Trained with Dataset    | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS                   | 0.99                   | 0.76                   | 0.77                  | 0.18                  |
| MNIST                   | 0.85                   | 0.99                   | 0.77                  | 0.22                  |
| ORHD                    | 0.39                   | 0.31                   | 0.98                  | 0.11                  |
| SVHN                    | 0.32                   | 0.65                   | 0.68                  | 0.93                  |

| Trained with 2 Dataset  | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS+MNIST             | 0.98                   | 0.99                   | 0.85                  | 0.24                  |
| ARDIS+ORHD              | 0.99                   | 0.79                   | 0.99                  | 0.23                  |
| ARDIS+SVHN              | 0.99                   | 0.85                   | 0.72                  | 0.93                  |
| MNIST+ORHD              | 0.90                   | 0.99                   | 0.99                  | 0.24                  |
| MNIST+SVHN              | 0.86                   | 0.99                   | 0.72                  | 0.94                  |
| ORHD+SVHN               | 0.48                   | 0.71                   | 0.98                  | 0.93                  |

| Trained with 3 Dataset  | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS+MNIST+ORHD        | 0.98                   | 0.99                   | 0.99                  | 0.24                  |
| ARDIS+MNIST+SVHN        | 0.98                   | 0.99                   | 0.67                  | 0.93                  |
| ARDIS+ORHD+SVHN         | 0.98                   | 0.82                   | 0.99                  | 0.90                  |
| MNIST+ORHD+SVHN         | 0.87                   | 0.99                   | 0.99                  | 0.93                  |

| Trained with 4 Dataset  | Test Accuracy on ARDIS | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :---------------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
| ARDIS+MNIST+ORHD+SVHN   | 0.99                   | 0.99                   | 0.99                  | 0.94                  |
