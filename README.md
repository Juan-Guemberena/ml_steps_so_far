# ml_steps_so_far
Show and give instructions to reproduce what we have already done so far


# OpenAI

## Install instructions

Get your API key from [here](https://beta.openai.com/docs/developer-quickstart/your-api-keys).
Set it as an env variable or store it with a secret management service.

Run

```
pip install openai
```
and then you can use it in your code like this:

```
import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(engine="davinci", prompt="This is a test", max_tokens=5)
```
This should return something like:
```
{
    "id": "cmpl-GERzeJQ4lvqPk8SkZu4XMIuR",
    "object": "text_completion",
    "created": 1586839808,
    "model": "davinci:2020-05-03",
    "choices": [{
        "text": " of reading speed. You",
        "index": 0,
        "logprobs": null,
        "finish_reason": "length"
    }]
}
```

You can also make cURL requests, like this (replacing YOUR_API_KEY with the proper key):
```
curl https://api.openai.com/v1/engines/davinci/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_API_KEY" \
-d '{"prompt": "This is a test", "max_tokens": 5}'
```
## [API Reference](https://beta.openai.com/docs/api-reference/introduction)

The first thing to note is that OpenAi provides different [engines](https://beta.openai.com/docs/engines), Davinci being the highest performing one, and Ada the fastest and cheapest.

They provide access to the API in the form of tokens, with 100 tokens approximately being 400 characters of text.

#### List of engines and their best use-cases (based on their docs)

* Davinci. Good at: Complex intent, cause and effect, summarization for audience.
* Curie. Good at: Language translation, complex classification, text sentiment, summarization.
* Babbage. Good at: Moderate classification, semantic search classification.
* Ada. Good at: Parsing text, simple classification, address correction, keywords.

 
The second thing to note is that they have simplified hyper-parameter tuning a bit, with temperature and top_p being the most important parameters. [Read more](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)

Temperature: Lower temperatures make the model increasingly confident in its top choices, while temperatures greater than 1 decrease confidence. 0 temperature is equivalent to argmax/max likelihood, while infinite temperature corresponds to a uniform sampling. 

So we would set a low temperature for pre-defined or structured answers (aka yes or no, or sentiment answers), and a high temperature for creative applications (like convergence)

Top P sampling (nucleus sampling): compute the cumulative distribution and cut off as soon as the CDF exceeds P. This helps avoiding egregiously wrong tokens while also preserving variety when the highest scoring tokens have low confidence.


The third thing to note is that OpenAI provides 2 more engines, called [Instruct series](https://beta.openai.com/docs/engines/the-instruct-series-beta), which can take, understand and follow a series of instructions. This can be used to create a product like the one seen [here](https://twitter.com/sharifshameem/status/1282676454690451457).

The last thing to note is that they also provide a [Content Filter](https://beta.openai.com/docs/engines/content-filter), which can be used to detect and avoid profanity or strong language. The filter classifies generated text in 3 categories: safe, sensitive or unsafe. They even provide an example Python code that checks the label returned by the filter as well as the logprobs, to thoroughly check that the returned label is correct.

With all of this, we should be capable of quickly developing any idea we have and try it out either locally or with their [Playground](https://beta.openai.com/playground), which unfortunately uses your token quota as well.




# Word2Vec to train

[Docs](https://code.google.com/archive/p/word2vec/)

### Quick Start

* Download the code: svn checkout http://word2vec.googlecode.com/svn/trunk/
* Run 'make' to compile word2vec tool
* Run the demo scripts: ./demo-word.sh and ./demo-phrases.sh
* For questions about the toolkit, see http://groups.google.com/group/word2vec-toolkit

After this, the word2vec tool can be used to train models.


# GloVe

[Docs](https://nlp.stanford.edu/projects/glove/)

[Github](https://github.com/stanfordnlp/GloVe)

### Getting started (Code download)

* Download the latest latest code (licensed under the Apache License, Version 2.0).
* Look for "Clone or download"
* Unpack the files:  unzip master.zip
* Compile the source:  cd GloVe-master && make
* Run the demo script: ./demo.sh
* Consult the included README for further usage details, or ask a question

### Download pre-trained word vectors

**Already tried the Twitter and Wikipedia word-vectors.**

* Pre-trained word vectors. This data is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/.
* Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
* Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): [glove.42B.300d.zip](http://nlp.stanford.edu/data/glove.42B.300d.zip)
* Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)
* Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): [glove.twitter.27B.zip](http://nlp.stanford.edu/data/glove.twitter.27B.zip)


# Gensim (develop word embeddings using word2vec and GloVe)

[Docs](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)


* pip install --upgrade gensim

Then you can use gensim in python to do things like 

```
sentences = ... # Matrix of sentences divided into words (e.g [['hello', 'world'],['how', 'are', 'you']])
model = Word2Vec(sentences)
```

### Example with pre-trained word2vec model trained on Google News data (3 mil words, 300-dimensional vectors)

* Download [the model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
* Unzip it
* Run the example
```
from gensim.models import KeyedVectors
# load the google word2vec model
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'],topn=1)
print(result)
```


### Example with GloVe pre-trained word vectors

* Download [the vectors](http://nlp.stanford.edu/data/glove.6B.zip)
* Unzip
* Load the model in word2vec format
```
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
```

* Run the example
```
from gensim.models import KeyedVectors
# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['woman', 'king'], negative=['man'],topn=1)
print(result)
```


# What we tried

* Training a model on glove-wiki-gigaword-300 (note that this uses 12 workers (processor threads) and has 300 dimensions, so training takes very long, especially since it has to do 5 epochs (see "iter" parameter))

[test.py](test.py)
```
import gensim.downloader as api
from annoy import AnnoyIndex
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus, LineSentence

LOGS = True
if LOGS:
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


corpus_path = api.load('glove-wiki-gigaword-300', return_path=True) # The model can also be loaded from disk, just not with this command


# Using params from Word2Vec_FastText_Comparison
params = {
    'alpha': 0.05,
    'size': 300,
    'window': 5,
    'iter': 5,
    'min_count': 5,
    'sample': 1e-4,
    'sg': 1,
    'hs': 0,
    'negative': 5,
    'workers':12
}
model = Word2Vec(LineSentence(corpus_path), **params)
model.save('glove-wiki-gigaword-300-model.model')
```


* Running a most_similar query on the GloVe model
 
What takes the most time is loading the model. After that, the most_similar query takes very little. We wonder if there is a way to keep the model loaded and query it multiple times, but that requires more memory the bigger the model. Read: [How to speed up Gensim Word2vec model load time?](https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time/43067907)

[txt_to_word2vec.py](txt_to_word2vec.py)
```
from gensim.scripts.glove2word2vec import glove2word2vec
# glove_input_file format example 'glove.6B.100d.txt'
glove_input_file = ''
word2vec_output_file = glove_input_file + ".word2vec"
glove2word2vec(glove_input_file, word2vec_output_file)
#Use the new output in most_similar.py
```

[most_similar.py](most_similar.py)
```
from gensim.models import KeyedVectors
# filename format should be for example 'glove.6b.100d.txt.word2vec'
filename = ''
model = KeyedVectors.load_word2vec_format(filename, binary=False)
print("Model loaded")
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['king','woman'],negative=['man'], topn=5)
print(result)
```

## Other models [here](https://github.com/RaRe-Technologies/gensim-data#models)


# Read

### [Understanding Word vectors](https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469)
### Annoy 
* [Annoy Approximate Nearest Neighbors Oh Yeah](https://github.com/spotify/annoy)
* [Similarity Queries with Annoy and Word2Vec](https://radimrehurek.com/gensim_3.8.3/auto_examples/tutorials/run_annoy.html)
* [ConceptNet](http://conceptnet5.media.mit.edu/)
