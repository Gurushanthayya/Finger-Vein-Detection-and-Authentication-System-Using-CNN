"""
CNN Model Architecture for Finger Vein Authentication (Month 2).
Defines a classification network and an embedding sub-network feature extractor.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

def create_classification_model(num_classes, input_shape=(64, 128, 1)):
    """
    Constructs the Classification CNN network.
    Input Layer -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten -> Dense -> Dropout -> Embedding -> Softmax.
    """
    inp = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 4
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    
    # Embedding projection layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation=None, name='embedding_dense')(x)
    
    # L2 normalize embeddings layer
    l2_embedding = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1), 
        name='l2_embedding'
    )(x)
    
    # Softmax Classification output
    out = layers.Dense(num_classes, activation='softmax', name='classifier_output')(l2_embedding)
    
    return models.Model(inputs=inp, outputs=out, name='vein_classification_network')


def extract_embedding_model(classification_model):
    """
    Extracts the feature embedding sub-network from the trained classification model.
    Cuts off the classification layer, returning the L2-normalized 128-d output.
    """
    # Find the output of the L2 normalization layer
    embedding_layer_output = classification_model.get_layer('l2_embedding').output
    return models.Model(inputs=classification_model.input, outputs=embedding_layer_output, name='vein_cnn_base')
