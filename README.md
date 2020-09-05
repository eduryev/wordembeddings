# WordEmbeddings

## Introduction
This is joint work of Efim Abrikosov and Eduard Duryev.

The goal of the project is to provide an efficient and easily accessible implementation of some classic word embeddings frameworks. In addition, we attempt to modify these frameworks and obtain new models driven by certain theoretical mathematical foundations.

The code contains following functional components:
1. Infrastructure to download, process and feed to models Wikipedia-based text corpora
2. Model definitions and training loop backed by tensorflow Keras model class
3. Similarity and analogy tests hooked to the training process via Tensorboard and test results displayed during the training time

This project is integrated with Google Cloud Platform and can be run on virtual machines through GCP `ai-platform`

## Instructions

The code can be used in three different ways depending on user's needs and preferences:
1. From command line by executing python script
```bash
python -m newmodel.task --job-dir=test_dir \
  --mode=glove \
  --embedding-size=200 \
  --corpus-name=enwik9 \
  --max-vocabulary-size=150000 \
  --min-occurrence=10 \
  --skip-window=10 \
  --num-epochs=50 \
```
2. Interactively with *WordEmbeddings_full.ipynb* notebook
   - Full notebook contains all declarations of classes and functions used in the training process. Best for experimenting and modifying the code
3. Semi-interactively with *WordEmbeddings_concise.ipynb* notebook
   - Concise notebook imports all necessary functions from the package. Best for testing models interactively

#### Word embeddings in 15 minutes

To quickly test the code *WordEmbeddings_full.ipynb* notebook can be launched in [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true). This won't require any installations or other changes to your local environment. The notebook code has many explanatory comments and contains a section "Training demos".

To train your own model, simply compile all cells preceding "Training demos", and then run cells for one of the examples. Please, note installation prompts in "Setup" section.

For instance, training a toy 200 dimensional Glove Model for 10 epochs should take about 15 minutes (including the time for the initial data processing). Real models can be trained in a couple of hours using larger datasets.


#### Google Cloud Platform integration

Word embedding models can be trained on Google Cloud Platform with different configurations of virtual machines. Some familiarity with GCP is required, in particular, understanding of control access.

If settings are set up properly and Docker container created with repo Dockerfile and tagged `$IMAGE_URI` is functional, the following analog of python module execution suffices to train models:

```bash
gcloud ai-platform jobs submit training $JOB_NAME \
  --config config.yaml \
  --region $REGION \
  --master-image-uri $IMAGE_URI \
  -- \
  --job-dir=$BUCKET_NAME/$MODEL_DIR \
  --log-dir=glove_model_test \
  --save-dir=glove_model_test \
  --mode=glove \
  --embedding-size=200 \
  --corpus-name=enwik9 \
  --max-vocabulary-size=150000 \
  --min-occurrence=10 \
  --skip-window=10 \
  --num-epochs=50 \
```


## Training frameworks
#### Model Types
Current version supports three model types. Each model is derived from `BaseModel` class defined in *newmodel.model* which is a child of `tf.keras.models.Model`. Thus all of the models inherit broadly used and efficient methods from their base class.
1. Glove Model `--mode=glove` ([Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/))

   Defined in`GloveModel` class
2. HypGlove Model `--mode=hypglove` &mdash; extension of Glove Model with an additional dimension for each word embedding (motivated by how "generic" is an embedded word)

  Defined in `HypGloveModel` class in *newmodel.model*
3. Word2Vec Model `--mode=word2vec` ([Tomas Mikolov et al. DistributeµÂtd Representations of Words and Phrases and Their Composability](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf))

  Defined in `Word2VecModel` class in *newmodel.model*

Embedding size for each model can be controlled through `embedding-size` parameter.

#### Supported datasets
The code is configured to download text corpora based on English Wikipedia. Text processing functions, such as skip counting can be applied more generally.

Data loading infrastructure is wrapped in higher-level functions such that `tf.data.Dataset` instance can be created in a few lines and passed immediately to `fit` method of a model (see examples in *WordEmbeddings_full.ipynb* notebook).

Processed data consists of triples: `(i1, i2, c)` where `i1` and `i2` are indices of two words in the vocabulary (ordered by frequency) and `c` is their "cooccurrence weight".

First two text corpora are downloaded from [Matt Mahoney's webpage](http://mattmahoney.net/dc/textdata.html).

1. Enwik8 `corpus-name=enwik8` (~ 36MB in zip format). Smallest dataset handy for quick prototyping and debugging
2. Enwik9 `corpus-name=enwik9` (~323MB in zip format). Larger dataset that typically offers considerable improvement over `enwik8`
3. English Wikipedia `corpus-name=enwiki_dump` &mdash; [full collection of current Wikipedia articles](https://dumps.wikimedia.org/). Current code processes fifteen bz2 files (~1.5GB of text data). Processing takes several hours and is computationally expensive. Some understanding of the process is recommended to avoid unintended big memory allocations

Additional parameters are passed when text is processed:
- `max_vocabulary_size` caps the number of words that will be tokenized for training. Remaining words are replaced with unknown token
- `min-occurrence` specifies minimal number of times a word should occur in a corpus to get a token assigned
- `skip-window` controls the maximal distance between two words in a corpus to contribute to their cooccurrence weight (1/distance weight is applied)

#### Training parameters
User can specify `--batch-size` and `--learning-rate`. Please note that learning rate is set to decay after 10th epoch through a callback (see `lr_scheduler_factory` function in *newmodel.util* for details)


## Training output
Model training process produces (1) reusable model weights and (2) training logs that can be viewed in Tensorboard. In addition, the class `BaseModel` provides various methods to explore embeddings (see e.g. `BaseModel.get_embedding`, `BaseModel.get_closest`).

Every training process affects only files inside a job directory passed through the required `job_dir` argument. This can be either a local path on your computer, similarly a path to a Google Drive destination, or a path in a Google Storage Bucket (for the latter two options appropriate access permissions are required).

Typically files inside the job directory will be organized as follows:
```
job_dir
   ├── logs
   ├── model_data
   ├── saved_models
   └── tests
         ├── analogy_tests
         └── similarity_tests
```

Model weights can be saved/restored to/from `saved_models` folder if `save_dir`/`restore_dir` argument is supplied. Similarly, logs will be stored inside `logs` folder if `log_dir` argument is supplied (defaults to `temp` subdirectory).

Tensorboard session with losses and test results can be launched in the usual manner from shell or using inline `%tensorboard` magic in .ipynb notebook, e.g.:
```bash
tensorboard --logdir logs/ --port=6006
```
