import json

from functools import partial
from bs4 import BeautifulSoup


class DataProcessor:
    DEFAULT_PARSER = "html.parser"
    DEFAULT_CFN = "words"
    DEFAULT_ENCODING = "ascii"
    DEFAULT_ERROR_BEHAVIOR = "ignore"
    VALID_SPLITS = ["TRAIN", "TEST"]

    def __init__(
        self,
        split_field,
        word_fields,
        label_fields,
        document_root,
        processors,
        corpus_field_name=DEFAULT_CFN,
        default_parser=DEFAULT_PARSER,
        default_encoding=DEFAULT_ENCODING,
        default_error_behavior=DEFAULT_ERROR_BEHAVIOR,
    ):
        """
        A generic data processor class that works on generic datasets.
            Receives a list of processors and runs them (Sequentially) on the parsed input data.
            Also parses SGML (XML in general) files using BeautifulSoup.

        returns a list of json_formatted parsed documents.

        :param split_field: The name of the field specifying whether
            each sample belongs to the training set or the test set
        :param word_fields: The name of the fields that comprise our whole vocabulary and we'd
            like to consider them as part of the final natural language corpus
        :param label_fields:  The name of the fields that can act as the label for our classification task
            They can also act as extra features to improve other label's classification accuracy
        :param document_root: The root XML element of each row
        :param processors: The processors working on each document's text to convert it to our desired state
        :param corpus_field_name: values of word_fields will be aggregated and returned under this name in
            a new document
        :param default_parser: The parser backend BeautifulSoup uses to parse input data
        :param default_encoding: The default encoding of the dataset of the text
        :param default_error_behavior: The default behavior in case a non-compliant chunk of text came
            up during processing
        """
        self.split_field = split_field
        self.word_fields = word_fields
        self.label_fields = label_fields
        self.document_root = document_root

        self.corpus_parser = default_parser
        self.corpus_encoding = default_encoding
        self.encoding_error_behavior = default_error_behavior

        self.corpus_field = corpus_field_name
        self._processors = processors or []

    def process(self, content):
        """
        Receives the content from apache beam's process, decodes the input data and collects the desired data
        according to how the class was configured to look for the data

        Finally, dumps the collected data into json formatted and returns the newly dumped data
        :param content:
        :return:
        """
        data = []
        root_parser = BeautifulSoup(
            content.read().decode(
                self.corpus_encoding, errors=self.encoding_error_behavior
            ),
            self.corpus_parser,
        )
        for row_parser in root_parser.find_all(self.document_root):
            get_values = partial(self.get_values, row_parser)
            corpus_values = []
            for key in self.word_fields:
                corpus_values += get_values(key)

            split_by = row_parser.attrs.get(self.split_field, "")
            if split_by not in self.VALID_SPLITS:
                continue
            document = {
                self.split_field: split_by,
                self.corpus_field: " ".join(sum(corpus_values, [])),
            }
            for label in self.label_fields:
                document.update({label: ",".join(get_values(label, process=False))})
            data.append(json.dumps(document))
        return data

    def get_values(self, parser, node_name, process=True):
        """

        :param parser: the parent node of the desired node
        :param node_name: name of the desired node, to enable to parent to find the child
        :param process: indicator flag, whether to run the processing logic on this value or not (Default: True)
        :return: List of all desired child nodes (possibly a list itself), or empty list
        """
        values = []
        for item in parser.find(node_name) or []:
            value = getattr(item, "text", str(item))
            if process:
                for processor in self._processors:
                    value = processor(value)
            values.append(value)
        return values
