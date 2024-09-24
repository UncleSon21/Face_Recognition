import tensorflow as tf
from tensortflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Layer

def make_embedding():
    input_shape = Input(shape=(100, 100, 3), name='input_image')
    
    c1 = Conv2D(64, (10,10), activation='relu')(input_shape)
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    
    c3 = Conv2D(128, (7,7), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    model = Model(inputs=[input_shape], outputs=[d1], name='embedding')
    
    return model

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    
def siamese_model():
    input_image = Input(name='input_img', shape=(100,100,3))
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    embedding = make_embedding()
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)
    
    siamese_layer = L1Dist()
    
    distances = siamese_layer(inp_embedding, val_embedding)
    
    classifier = Dense(1, activation='sigmoid')(distances)
    
    siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
    
    return siamese_network