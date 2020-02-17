# Getting Started

There are several ways to get started with the project.

There's a simple Dockerfile bundled with the project to make running the project easier. Build a docker image in the
 root of the project (where you can find the `Dockerfile`) and wait for the image build to finish.
Once finished, you can run the project using

```shell script
docker build . -t whatever
docker run -it whatever --dataset-dir=reuters/dataset
```
If you don't want to go the docker way, just install the dependencies in a virtual environment, or your global python
 installation and after it's done, run the project, the main script to run is `starter.py`.
 
```shell script
pip install -r requirements/dev.txt
python starter.py --dataset-dir=reuters/dataset
```

Just have in mind that the first time you run the project, it's gonna take sometime to pre-process all the raw data. 
It will be cached later on and subsequent runs of the model should be relatively speedy. You can try switching 
Apache Beam's `DirectRunner` for a `DataflowRunner` and see how much it can speed data pre-processing up.

Dataset will be downloaded on the first use.

## How to run tests
Run the following in the root of the project. Testing for now is _very_ limited.
```shell script
python -m unittest discover
```

# About
1. Initially all data passes through several steps in Beam. Doing so features are transformed in a distributed fashion,
and can later be put to production to handle billions of rows of data, without requiring us to re-write 
pre-processing steps anew.
Also TF-Transform can be easily employed to remove train-test skew in realtime or batch loads in huge scale, 
since we're using beam.
2. To parse SGML format, a python library called `BeautifulSoup` is used.
3. One-hot-encoded vectors are used for labels, while an `embedding` is learned for the input texts.
4. We do multi-label classification, running the training algorithm for 1 epoch reaches a top_5 accuracy of
 nearly +83% which is fine, it can certainly improve with more epochs.
5. A 2 layer stacked LSTM model is used to classify the data, with 2 Dropout layers to reduce over-fitting.
6. Generally, solutions here might not be super amazing since it was a showcase of engineering + problem solving
 and time was short! (e.g. the `logging` module, which is setup but not fully working and I didn't find anytime
  to take a look, have backtracked to print where necessary)    

# Future Directions 
* Train for more epochs, and on better infrastructure (e.g. GPUs, GCP AI Platform, ...) 
* Use better word/sentence embeddings pre-trained on larger outsider datasets, like word2vec or
 Universal Sentence Encoder
* Use a better recurrent model architecture, probably with windowing
* Test out other models as well, 1D CNNs might also be good fits for this task, TfIDF or other relevant ideas
* Use features like "places" or other relevant information in the dataset to improve classification
* Use BayesianOptimization or a GridSearch to figure out a good set of hyperparameters for the models and vocabularies

# Contact
I usually am instantaneously available by email, `alirezasadeghi71@gmail.com`!