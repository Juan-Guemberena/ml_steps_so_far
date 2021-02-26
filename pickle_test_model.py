import pickle
# filename should be .sav format
filename = ''
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.most_similar(positive=[],negative=[], topn=5)
print(result)