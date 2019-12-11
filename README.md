# CNN Digit Recognition

Simple application for digit recognition with CNN using three different datasets.

 * MNIST Database of Handwritten Digits [link](https://keras.io/datasets/)

 * ORHD - Optical Recognition of Handwritten Digits Data Set [link](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)

 * SVHN - Street View House Numbers Cropped Digit Dataset [link](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_House_Numbers_%28SVHN%29_Dataset)

## Usage

Project models created in virtual environment using [anaconda](https://www.anaconda.com/).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

This environment generated with tensorflow gpu!

```sh
conda env create -f environment.yml
```

Then you can train models with.

```python
python trainer.py
```

## Experimental Results

| Trained with Dataset | Test Accuracy on MNIST | Test Accuracy on ORHD | Test Accuracy on SVHN |
| :------------------: | :--------------------: | :-------------------: | :-------------------: |
| MNIST                | 0.99                   | 0.72                  | 0.28                  |
| ORHD                 | 0.25                   | 0.97                  | 0.13                  |
| SVHN                 | 0.59                   | 0.45                  | 0.90                  |     
| ORHD+MNIST           | 0.99                   | 0.98                  | 0.28                  |
| ORHD+SVHN            | 0.60                   | 0.97                  | 0.89                  |
| SVHN+MNIST           | 0.99                   | 0.64                  | 0.90                  |
| ORHD+SVHN+MNIST      | 0.99                   | 0.98                  | 0.90                  |
