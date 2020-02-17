import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer


class Tokenizer(object):
    """
    Takes care of one-hot-encoding the labels, tokenizing, converting to sequences and padding of the input data
    """

    def __init__(self, dataframe):
        self.df = dataframe

    def tokenize(self):
        _train_df = self.df[self.df.lewissplit == "TRAIN"]
        _test_df = self.df[self.df.lewissplit == "TEST"]

        X_train, Y_train = _train_df["words"], _train_df["topics"]
        X_test, Y_test = _test_df["words"], _test_df["topics"]

        label_1hot_encoder = CountVectorizer(
            tokenizer=lambda x: x.split(","), binary="true"
        )
        Y_train = label_1hot_encoder.fit_transform(Y_train).toarray()
        Y_test = label_1hot_encoder.transform(Y_test).toarray()

        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(X_train)

        X_train = tf.keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences(X_train),
            maxlen=200,
            padding="pre",
            truncating="pre",
        )

        X_test = tf.keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences(X_test),
            maxlen=200,
            padding="pre",
            truncating="pre",
        )

        return (X_train, Y_train), (X_test, Y_test), len(tokenizer.word_index)
