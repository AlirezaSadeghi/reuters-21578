import tensorflow as tf

from datetime import datetime


class BaseClassifier(object):
    """
    Interface for all classifiers
    """

    def __init__(self, x_train, y_train, vocab_size):
        self.X = x_train
        self.Y = y_train
        self.N = vocab_size


class RecurrentClassifier(BaseClassifier):
    """
    Trains a Recurrent Neural Network (2 stacked LSTMs on top of each other here) and returns the trained model

    TODO: API here is neither great nor customizable
    """

    def __init__(self, x_train, y_train, vocab_size):
        super().__init__(x_train, y_train, vocab_size)
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    self.N + 1, output_dim=50, input_length=self.X.shape[1]
                ),
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(116, activation="sigmoid"),
            ]
        )

    def train(self):
        artifacts_dir = f"reuters/artifacts/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "top_k_categorical_accuracy"],
        )
        self.model.fit(
            self.X,
            self.Y,
            class_weight="balanced",
            epochs=1,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    f"{artifacts_dir}/checkpoints/", save_best_only=True
                ),
                tf.keras.callbacks.TensorBoard(log_dir=f"{artifacts_dir}/tensorboard/"),
            ],
        )

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x):
        return self.model.predict(x)


class ConvolutionalClassifier(BaseClassifier):
    pass


class NaiveBayesClassifier(BaseClassifier):
    pass
