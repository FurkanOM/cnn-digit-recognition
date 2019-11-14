from keras.models import load_model
import matplotlib.pyplot as plt
import Helpers

MNIST = Helpers.get_MNIST_data()
SVHN = Helpers.get_SVHN_data()
#
models = Helpers.get_files("./models")
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for modelData in models:
    trained_with = modelData["filename"].split("_")[0]
    print("Model loaded, trained with:", trained_with, "dataset")
    model = load_model(modelData["path"])
    #
    y_predicted = model.predict(MNIST["x_test"])
    actuals = MNIST["y_test"].argmax(axis=1)
    predictions = y_predicted.argmax(axis=1)
    title = "MNIST confusion matrix by model with trained " + trained_with
    save_path = "./results/mnist_cm_by_" + trained_with.lower() + ".png"
    Helpers.handle_confusion_matrix(actuals, predictions, labels, title, save_path=save_path)
    #
    y_predicted = model.predict(SVHN["x_test"])
    actuals = SVHN["y_test"].argmax(axis=1)
    predictions = y_predicted.argmax(axis=1)
    title = "SVHN confusion matrix by model with trained " + trained_with
    save_path = "./results/svhn_cm_by_" + trained_with.lower() + ".png"
    Helpers.handle_confusion_matrix(actuals, predictions, labels, title, save_path=save_path)
    print("============================================================================")

plt.show()
