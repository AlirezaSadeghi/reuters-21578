import os
import nltk
import tarfile
from urllib.request import urlretrieve


class ReutersUtils(object):
    DATASET_URI = "http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz"

    @staticmethod
    def find_file(path, name_prefix):
        for name in os.listdir(path):
            if name.startswith(name_prefix):
                return os.path.join(path, name)
        return None

    @staticmethod
    def download_nltk_packages():
        nltk.download("punkt")
        nltk.download("stopwords")

    @staticmethod
    def ensure_dataset(path):
        if os.path.isdir(path) and len(os.listdir(path)) > 20:
            return True
        print("Dataset Not Found | Downloading and Extracting Dataset")
        if not os.path.isdir(path):
            os.mkdir(path)
        name = f"{path}/dataset.tar.gz"
        urlretrieve(ReutersUtils.DATASET_URI, name)
        tarfile.open(name).extractall(path)
