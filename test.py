import gensim.downloader as api
from annoy import AnnoyIndex
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus, LineSentence

LOGS = True
if LOGS:
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


corpus_path = api.load('glove-wiki-gigaword-300', return_path=True)


# Using params from Word2Vec_FastText_Comparison
params = {
    'alpha': 0.05,
    'size': 300,
    'window': 5,
    'iter': 1,
    'min_count': 5,
    'sample': 1e-4,
    'sg': 1,
    'hs': 0,
    'negative': 5,
    'workers':12
}
model = Word2Vec(LineSentence(corpus_path), **params)
model.save('glove-wiki-gigaword-300-model.model')




