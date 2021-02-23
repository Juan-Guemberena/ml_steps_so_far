from gensim.models import KeyedVectors
# filename format should be for example 'glove.6b.100d.txt.word2vec'
filename = ''
model = KeyedVectors.load_word2vec_format(filename, binary=False)
print("Model loaded")
# calculate: (king - man) + woman = ?
result = model.most_similar(positive=['king','woman'],negative=['man'], topn=5)
print(result)