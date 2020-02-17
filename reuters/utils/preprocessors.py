import nltk
import string

from nltk.stem.porter import *
from nltk.corpus import stopwords


class BasePreprocessor:
    """
    Base preprocessor, defining the interface for our common preprocessors
    """

    def __call__(self, value):
        raise NotImplementedError("Should be implemented")


class LowercasePreprocessor(BasePreprocessor):
    """
    Lower-cases Strings
    """

    def __call__(self, value):
        return value.lower()


class NoDigitsPreprocessor(BasePreprocessor):
    """
    Removes digits from any String
    """

    def __call__(self, value):
        return value.translate(str.maketrans("", "", string.digits))


class NoPunctuationPreprocessor(BasePreprocessor):
    """
    Removes punctuation from Strings. Stuff like ".", ",", ":", "'" and the rest
    """

    def __call__(self, value):
        return value.translate(str.maketrans("", "", string.punctuation))


class TokenizerPreprocessor(BasePreprocessor):
    """
    Tokenize arbitrary sized sequence of words
    """

    def __call__(self, value):
        return nltk.word_tokenize(value)


class NoStopWordPreprocessor(BasePreprocessor):
    """
    Removes stop-words from a text sequence, since they usually don't add much value to the final
    text representation.

    The input to this stage should be a sequence, not a text, so it converts its input to a Seq otherwise
    """

    def __init__(self, extra_words=None):
        self.extra_words = extra_words or []

    def __call__(self, value):
        if not isinstance(value, list):
            value = TokenizerPreprocessor()(value)
        return [
            item
            for item in value
            if item not in (stopwords.words("english") + self.extra_words)
        ]


class StemmingPreprocessor(BasePreprocessor):
    """
    Keeps only a stemmed version of all words seen to reduce duplication of close concepts in different forms
    Only works for ascii encoding and ignores error cases
    The input to this stage should be a sequence, not a text, so it converts its input to a Seq otherwise
    """

    MIN_LENGTH = 4

    def __call__(self, value):
        if not isinstance(value, list):
            value = TokenizerPreprocessor()(value)
        stems = []
        stemmer = PorterStemmer()
        for token in value:
            stem = stemmer.stem(token)
            if len(stem) >= StemmingPreprocessor.MIN_LENGTH:
                stems.append(stem)
        return stems


string_processors = [
    LowercasePreprocessor(),
    NoDigitsPreprocessor(),
    NoPunctuationPreprocessor(),
    TokenizerPreprocessor(),
]
