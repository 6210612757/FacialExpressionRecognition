from model_training import split_xy

import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2 as cv
import seaborn as sns
import tabulate as tabulate

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import multilabel_confusion_matrix

# 21

MODEL_PATH = '\\model\\cp-0025.ckpt'


test_internet = "C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\internet_test"
TEST_PATH = "C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\dataset\\test"
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
img_size = (224, 224)


def main():
    data_test = pd.read_csv("fer2013/fer2013.csv", engine='python',
                            encoding='utf-8', error_bad_lines=False)
    data_test.query(
        'Usage =="PublicTest" or Usage =="PrivateTest"', inplace=True)
    X_test = [np.fromstring(
        x, dtype=int, sep=' ').reshape(-1, 48, 48) for x in data_test['pixels']]
    X_test = np.array(X_test).reshape(-1, 48, 48).astype('uint8')
    print(X_test.shape)
    # Make data rgb 3 channels to make model accept 3 channels
    X_test = np.repeat(X_test[..., np.newaxis], 3, -1)

    X_test = np.asarray(
        [cv.resize(x, img_size, interpolation=cv.INTER_LINEAR) for x in X_test])

    Y_test = data_test.emotion.values
    print("Test dataset:")
    print(X_test.shape)
    print(Y_test.shape)

    model = create_model()
    model.load_weights(MODEL_PATH)

    evaluate_model("FER", model, X_test, Y_test, emotions)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PATH,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        class_names=emotions,
        color_mode='rgb',
        batch_size=32)
    show_img(test_ds, emotions)
    x_test, y_test = split_xy(test_ds)
    evaluate_model("regular", model, x_test, y_test, emotions)

    # in_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #     test_internet,
    #     labels='inferred',
    #     label_mode='int',
    #     image_size=img_size,
    #     class_names=emotions,
    #     color_mode='rgb',
    #     batch_size=32)

    # x_test, y_test = split_xy(in_test_ds)
    # evaluate_model("regular", model, x_test, y_test, emotions)


def evaluate_model(name, model, x_test, y_test, labels):

    print("=======================================")
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=2)
    print(f'\nTest {name} loss:', test_loss)
    print(f'\nTest {name} accuracy:', test_acc)

    prediction = model.predict(x_test)
    y_pred = np.argmax(prediction, axis=1)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    # print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=emotions)

    disp.plot()
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=emotions)

    disp.plot()
    plt.show()

    sns.heatmap(cm/np.sum(cm), annot=True,
                fmt='.2%', cmap='Blues')
    plt.show()

    my_classification_report = classification_report(
        y_test, y_pred, target_names=emotions, digits=4)
    print(my_classification_report)
    print("^^^above module Count true negative in accuracy^^^")
    print()
    print("============================")

    mcm = multilabel_confusion_matrix(
        y_test, y_pred, labels=range(len(emotions)))

    # print(mcm)
    metrics = ["Metrics", "Accuracy", "Sensitivity",
               "Specificity", "Precision", "NPV", "F1"]
    acc = []
    spec = []
    sen = []
    pre = []
    npv = []

    f1 = []
    for i in mcm:
        tn, fp, fn, tp = i.ravel()
        acc.append((tp+tn)/(tn+fp+fn+tp))
        spec.append(tn/(tn+fp))
        sen.append(tp/(tp+fn))
        pre.append(tp/(tp+fp))
        npv.append(tn/(fn+tn))

    for i in range(len(labels)):
        f1.append((2*sen[i]*pre[i])/(sen[i]+pre[i]))

    # print(labels)
    # print("Accuracy :", acc)
    print("Macro Accuracy:", np.mean(acc))
    # print("Sensitivity :", sen)
    print("Macro Sensitivity:", np.mean(sen))
    # print("Specificity :", spec)
    print("Macro Specificity:", np.mean(spec))
    # print("Precision :", pre)
    print("Macro Precision:", np.mean(pre))
    # print("Negative predictive value :", npv)
    print("Macro npv:", np.mean(npv))
    # print("F1-Score :", f1)
    print("Macro f1:", np.mean(f1))

    table = zip(emotions, acc, sen, spec, pre, npv, f1)

    print(tabulate.tabulate(table, headers=metrics, floatfmt=".4f"))
    print("=======================================")


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(img_size[0], img_size[1], 3)),
        # tf.keras.layers.experimental.preprocessing.RandomRotation(
        #     0.05, fill_mode='nearest'),

        tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(img_size[0], img_size[1], 3), include_preprocessing=True
        ),

        # tf.keras.layers.BatchNormalization(),

        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(256, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(7, activation='softmax')

    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    return model


def show_img(ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(len(class_names)):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


if __name__ == "__main__":
    main()
