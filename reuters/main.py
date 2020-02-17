import os
import logging
import pandas as pd

import apache_beam as beam
from apache_beam.io import WriteToText
from apache_beam.io.fileio import MatchFiles, ReadMatches

from reuters.models import RecurrentClassifier
from reuters.preprocessing import DataProcessor
from reuters.tokenizer import Tokenizer
from reuters.utils.preprocessors import (
    string_processors,
    NoStopWordPreprocessor,
    StemmingPreprocessor,
)
from reuters.utils.utility import ReutersUtils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROCESSED_FILE_PREFIX = "processed"


def transform_data(dataset_dir, name_pattern):
    """
    Spins up an apache beam runner (Here, DirectRunner is used since we're running it locally) to take care of
    pre-processing the dataset in a distributed manner.

    When complete, it writes out the processed data into a file prefixed "processed" (for now, not the best choice, but
    short on time and have to compromise)

    :param dataset_dir: The parent directory of the dataset
    :param name_pattern: Glob pattern of the input files in the dataset
    """
    processors = string_processors + [
        NoStopWordPreprocessor(extra_words=["reuter", "\x01", "\x03"]),
        StemmingPreprocessor(),
    ]
    data_processor = DataProcessor(
        split_field="lewissplit",
        word_fields=["title", "body"],
        label_fields=["places", "topics"],
        document_root="reuters",
        processors=processors,
    )

    with beam.Pipeline() as pipeline:
        _ = (
            pipeline
            | "Find Files" >> MatchFiles(os.path.join(dataset_dir, name_pattern))
            | "Read Files" >> ReadMatches()
            | "Map" >> beam.FlatMap(data_processor.process)
            | "Write"
            >> WriteToText(os.path.join(dataset_dir, PROCESSED_FILE_PREFIX + ".txt"))
        )


def main(args):
    ReutersUtils.ensure_dataset(args.dataset_dir)
    ds_path = ReutersUtils.find_file(args.dataset_dir, PROCESSED_FILE_PREFIX)

    while not ds_path:
        ReutersUtils.download_nltk_packages()
        transform_data(args.dataset_dir, "reut2-0*.sgm")
        ds_path = ReutersUtils.find_file(args.dataset_dir, PROCESSED_FILE_PREFIX)

    df = pd.read_json(ds_path, lines=True)

    (X_train, Y_train), (X_test, Y_test), vocab_size = Tokenizer(df).tokenize()

    classifier = RecurrentClassifier(X_train, Y_train, vocab_size)
    classifier.train()
    evaluation = classifier.evaluate(X_test, Y_test)

    print(
        f"LSTM Model: Loss: {evaluation[0]}, Accuracy: {evaluation[1]}, Top 5 Accuracy: {evaluation[2]}"
    )
