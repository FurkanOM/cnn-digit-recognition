from keras.models import load_model
import Helpers

MNIST = Helpers.get_MNIST_data()
SVHN = Helpers.get_SVHN_data()
#
models = Helpers.get_files("./models")
for modelData in models:
    trained_with = modelData["filename"].split("_")[0]
    print("Model loaded, trained with:", trained_with, "dataset")
    model = load_model(modelData["path"])
    #
    MNIST_score = model.evaluate(MNIST["x_test"], MNIST["y_test"], verbose=0)
    print("Trained with:", trained_with, "MNIST Test loss:", MNIST_score[0])
    print("Trained with:", trained_with, "MNIST Test accuracy:", MNIST_score[1])
    #
    SVHN_score = model.evaluate(SVHN["x_test"], SVHN["y_test"], verbose=0)
    print("Trained with:", trained_with, "SVHN Test loss:", SVHN_score[0])
    print("Trained with:", trained_with, "SVHN Test accuracy:", SVHN_score[1])
    print("============================================================================")

import code
code.interact(local=locals())
