from gensim.models import KeyedVectors
import pickle
#filename format should be for example 'glove.6b.100d.txt.word2vec'
filename = ''
model = KeyedVectors.load_word2vec_format(filename, binary=False)

#filename format should be for example 'glove.6b.100d.sav'
filename = ''
pickle.dump(model, open(filename, 'wb'))