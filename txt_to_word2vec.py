from gensim.scripts.glove2word2vec import glove2word2vec
# glove_input_file format example 'glove.6B.100d.txt'
glove_input_file = ''
word2vec_output_file = glove_input_file + ".word2vec"
glove2word2vec(glove_input_file, word2vec_output_file)
#Use the new output in most_similar.py