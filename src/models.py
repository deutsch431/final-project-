import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_tabular_mlp(input_shape, num_classes):
    """
    Builds a Multilayer Perceptron for tabular product metadata.
    Suitable for classification or regression depending on configuration.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax' if num_classes > 1 else None)
    ])
    return model

def build_vision_cnn(input_shape, num_classes):
    """
    Builds a custom CNN model from scratch for product image classification.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_vision_mobilenet(input_shape, num_classes):
    """
    Builds a transfer learning model using pre-trained MobileNetV2.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze backbone weights
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_text_embedding_model(vocab_size, embedding_dim, max_length, num_classes):
    """
    Builds a custom Keras text classifier using an embedding layer with strong regularization.
    """
    model = models.Sequential([
        layers.Input(shape=(max_length,)),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_regularizer=regularizers.l2(1e-4)
        ),
        layers.SpatialDropout1D(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax' if num_classes > 1 else None)
    ])
    return model
