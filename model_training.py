import pandas as pd
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import cv2
import tensorflow_addons as tfa

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
epochs = 300
batch_size = 32
img_size = (224, 224)


def main():

    # data_train = pd.read_csv("fer2013/fer2013.csv", engine='python',
    #                          encoding='utf-8', error_bad_lines=False)
    # data_train.query('Usage == "Training"', inplace=True)
    # X = [np.fromstring(
    #     x, dtype=int, sep=' ').reshape(-1, 48, 48) for x in data_train['pixels']]
    # X = np.array(X).reshape(-1, 48, 48)
    # # Make data rgb 3 channels to make model accept 3 channels
    # X = np.repeat(X[..., np.newaxis], 3, -1)
    # Y = data_train.emotion.values
    # print(X.shape)
    # print(Y.shape)

    data_test = pd.read_csv("fer2013/fer2013.csv", engine='python',
                            encoding='utf-8', error_bad_lines=False)
    data_test.query(
        'Usage =="PublicTest" or Usage =="PrivateTest"', inplace=True)
    X_test = [np.fromstring(
        x, dtype=int, sep=' ').reshape(-1, 48, 48) for x in data_test['pixels']]
    X_test = np.array(X_test).reshape(-1, 48, 48).astype('uint8')
    # Make data rgb 3 channels to make model accept 3 channels
    X_test = np.repeat(X_test[..., np.newaxis], 3, -1)
    X_test = np.asarray(
        [cv2.resize(x, img_size, interpolation=cv2.INTER_LINEAR) for x in X_test])

    Y_test = data_test.emotion.values
    print("Test FER dataset:")
    print(X_test.shape)
    print(Y_test.shape)

    print("Load train data")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\dataset\\train',
        shuffle=True,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        class_names=emotions,
        color_mode='rgb',
        batch_size=batch_size)
    x_train, y_train = split_xy(train_ds)

    print("Load validation data")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\dataset\\val',
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        class_names=emotions,
        color_mode='rgb',
        batch_size=batch_size)

    print("Load test data")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\dataset\\test',
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        class_names=emotions,
        color_mode='rgb',
        batch_size=batch_size)

    print("Load internet test data")
    rgb_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "C:\\Users\\User\\Desktop\\Project\\Depression\\FER\\internet_test",
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        class_names=emotions,
        color_mode="rgb",
        batch_size=batch_size)

    print("Train dataset:", x_train.shape)

    class_weights = dict(enumerate(class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train)))

    print(class_weights)
    # class_weights = {0: 2.0787244228887825, 1: 3.6863787375415282, 2: 3.958458855240209,
    #                  3: 0.44961505560307957, 4: 1.1183980647762397, 5: 1.5718692942139394, 6: 0.4463095796741972}

    # x_train, y_train = augment_concate(x_train, y_train, "triple")
    # print("After augment:")
    # print(x_train.shape)
    # print(y_train.shape)

    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=epochs*0.05, restore_best_weights=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model/cp-{epoch:04d}.ckpt",
        verbose=0,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True)
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir="./logs",
                update_freq="epoch")
    # # if some epochs no improve reduce learning rate by 20%
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_accuracy', factor=0.2, patience=epochs*0.02)

    model = create_model()

    print(model.summary())
    history = model.fit(train_ds, validation_data=val_ds, shuffle=True, epochs=epochs, callbacks=[es_callback, cp_callback, tb_callback
                                                                                                  # , reduce_lr
                                                                                                  ], class_weight=class_weights, batch_size=batch_size)

    print("============= Model Result at last epoch ==================")
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)

    f = open("testsplit.txt", "w")
    f.write('Test loss: {} \nTest accuracy: {}'.format(
        test_loss, test_acc))
    f.close()
    print("-------------")
    test_loss, test_acc = model.evaluate(x=X_test, y=Y_test, verbose=2)
    print('\nTest FER loss:', test_loss)
    print('\nTest FER accuracy:', test_acc)

    f = open("testFER.txt", "w")
    f.write('Gray Test loss: {} \nGray Test accuracy: {}'.format(
        test_loss, test_acc))
    f.close()
    print("-------------")
    test_loss, test_acc = model.evaluate(rgb_test_ds, verbose=2)
    print('\nTest Internet loss:', test_loss)
    print('\nTest Internet accuracy:', test_acc)

    f = open("testrgb.txt", "w")
    f.write('RGB Test loss: {} \nRGB Test accuracy: {}'.format(test_loss, test_acc))
    f.close()
    print("========================================")


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(img_size[0], img_size[1], 3)),
        tf.keras.layers.experimental.preprocessing.RandomRotation(
            0.05, fill_mode='nearest'
        ),

        tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(img_size[0], img_size[1], 3), include_preprocessing=True
        ),


        # tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.3),
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

    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True),  # tf.keras.optimizers.Adam(amsgrad=True),  # tf.keras.optimizers.SGD(nesterov=True), # tfa.optimizers.RectifiedAdam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    return model


def augment_concate(images, labels, augment: str = ''):
    # 1 full
    if augment == 'extra':
        new_images = np.concatenate((images, (jpeg_quality_augment(
            images), lighting_augment(images), rotate_augment(images), jpeg_quality_augment(rotate_augment(images)), lighting_augment(rotate_augment(images)))), axis=0)
        new_labels = np.concatenate(
            (labels, labels, labels, labels, labels, labels), axis=0)

    # 4 double
    elif augment == "double":
        new_images = np.concatenate((images, lighting_augment(images)), axis=0)
        new_labels = np.concatenate((labels, labels), axis=0)
    # 0,2,5 triple
    elif augment == "triple":
        new_images = np.concatenate((images, lighting_augment(
            images), jpeg_quality_augment(images)), axis=0)
        new_labels = np.concatenate((labels, labels, labels), axis=0)
    else:
        print("Error augment type")
    return new_images, new_labels


def rotate_augment(img):
    augmentrotate = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, fill_mode='nearest')
                                         ])
    return augmentrotate(img)


def jpeg_quality_augment(img):
    if len(img.shape) > 3:
        return np.array([tf.image.random_jpeg_quality(i, 10, 70) for i in img])
    else:
        return tf.image.random_jpeg_quality(img, 10, 70)


def lighting_augment(img):
    if len(img.shape) > 3:
        return np.array([tf.image.random_brightness(i, 0.3) for i in img])
    else:
        return tf.image.random_brightness(img, 0.3)


def hue_augment(img):
    if len(img.shape) > 3:
        return np.array([tf.image.random_hue(i, 0.1) for i in img])
    else:
        return tf.image.random_hue(img, 0.1)


def split_xy(data_set):
    # loop batch
    images = list()
    labels = list()
    for img_batch, label_batch in data_set:
        for i in range(len(img_batch)):
            images.append(img_batch[i].numpy().astype("uint8"))
            labels.append(label_batch[i].numpy().astype("uint8"))
    images = np.array(images)
    labels = np.array(labels)
    return images.squeeze(), labels.reshape(-1)


if __name__ == "__main__":
    main()
