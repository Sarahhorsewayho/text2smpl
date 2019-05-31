import numpy as np

from lib.config import cfg
from gensim.models.word2vec import Word2Vec


def word2index(caption_tuples):

	words = caption2words(caption_tuples)
	word2idx = {}
	idx2word = {}
	vocab_size = len(words)

	for i, word in enumerate(words):
		word2idx[word] = i
		idx2word[i] = word
	return word2idx, idx2word

def caption2words(caption_tuples):

	words = []
	for i in range(len(caption_tuples)):
		for word in caption_tuples[i].split():
			words.append(word)
	
	words = set(words)
	return words

def words2list(caption_tuples):

	idx_list = []
	mx_len = compute_max_length(caption_tuples)
	word2idx, idx2word = word2index(caption_tuples)
	for i in range(len(caption_tuples)):
		sentence = []
		for word in caption_tuples[i].split():
			sentence.append(word2idx[word])
		if len(sentence) < mx_len:
			for j in range(len(sentence), mx_len):
				sentence.append(0)
		idx_list.append(np.array(sentence))
	return idx_list

def gen_captions(caption_tuples):

	model = load_word2vec_model()
	max_length = compute_max_length(caption_tuples)

	captions_post = []

	for i in range(len(caption_tuples)):		
		caption = []	
		for word in caption_tuples[i].split():
			
			if word in model:
				caption.append(np.array(model[word]))
			else:
				caption.append([0] * cfg.CONST.vec_dim)

		if len(caption) < max_length:
			for j in range(len(caption), max_length):
				caption.append([0] * cfg.CONST.vec_dim)
		
		captions_post.append(caption)
	return captions_post

def caption2list(caption_tuples):	
	sentences = []

	for i in range(len(caption_tuples)):
		sentence = []
		for word in caption_tuples[i].split():
			#if(word != ','):
			sentence.append(word)
		sentences.append(sentence)

	return sentences

def word_to_vec(caption_tuples):

	sentences = caption2list(caption_tuples)
	model = Word2Vec(sentences, min_count=0, size=cfg.CONST.vec_dim)
	model.save(cfg.DIR.word2vec_model_path)

def load_word2vec_model():

	model = Word2Vec.load(cfg.DIR.word2vec_model_path)
	return model

def compute_max_length(caption_tuples):
	max_length = 0
	for i in range(len(caption_tuples)):
		cnt = 0
		for word in caption_tuples[i].split():
			cnt = cnt + 1
		if cnt >= max_length:
			max_length = cnt

	print("=========== max_length ==============")
	print(max_length)

	return max_length




