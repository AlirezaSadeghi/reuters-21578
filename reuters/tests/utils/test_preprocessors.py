import unittest

from reuters.utils.preprocessors import (
    LowercasePreprocessor,
    NoDigitsPreprocessor,
    NoPunctuationPreprocessor,
    NoStopWordPreprocessor,
    TokenizerPreprocessor,
    StemmingPreprocessor,
)


class TestPreprocessors(unittest.TestCase):
    def test_lower_case(self):
        """
        Test that it does lowercase input strings
        """
        data = "Variable Case inPutString."
        self.assertEqual(LowercasePreprocessor()(data), data.lower())

    def test_no_digits(self):
        """
        Test for removal of digits
        """
        data = "Shell193's stock is 31.51 per share"
        self.assertEqual(NoDigitsPreprocessor()(data), "Shell's stock is . per share")

    def test_no_punctuation(self):
        """
        Test for removal of punctuation
        """
        data = "He said: be here! Okay?:|"
        self.assertEqual(NoPunctuationPreprocessor()(data), "He said be here Okay")

    def test_no_stop_word(self):
        """
        Test for removal of stop ds
        """
        data = "He would do that, or should he?"
        self.assertEqual(NoStopWordPreprocessor()(data), ["He", "would", ",", "?"])

    def test_no_stemmings(self):
        """
        Test for removal of stemmings
        """
        self.assertListEqual(
            StemmingPreprocessor()("He's gonna go there and then be going back"),
            ["there", "then", "back"],
        )
        self.assertSetEqual(
            set(
                StemmingPreprocessor()(
                    "He distinguishes, always is actually distinguishing"
                )
            ),
            set(["distinguish", "alway", "actual"]),
        )

    def test_tokenizer(self):
        """
        Test for correct tokenization
        """
        data = "I will be there"
        self.assertListEqual(
            TokenizerPreprocessor()(data), ["I", "will", "be", "there"]
        )


if __name__ == "__main__":
    unittest.main()
