import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Layer


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


def siamese_model():
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    embedding = make_embedding()
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)

    # Ensure we're working with tensors, not lists
    if isinstance(inp_embedding, list):
        inp_embedding = inp_embedding[0]
    if isinstance(val_embedding, list):
        val_embedding = val_embedding[0]

    print("inp_embedding shape:", tf.keras.backend.int_shape(inp_embedding))
    print("val_embedding shape:", tf.keras.backend.int_shape(val_embedding))

    siamese_layer = L1Dist()
    distances = siamese_layer([inp_embedding, val_embedding])

    print("Distances shape:", tf.keras.backend.int_shape(distances))

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
