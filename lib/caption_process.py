import numpy as np

from lib.config import cfg
from gensim.models.word2vec import Word2Vec

VEC_DIM = 128

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

def gen_captions(caption_tuples):

	model = load_word2vec_model()
	max_length = compute_max_length(caption_tuples)

	captions_post = []

	for i in range(len(caption_tuples)):
		
		caption = []
		
		for word in caption_tuples[i].split():
			
			if word in model:
				caption.append(model[word])
			else:
				caption.append([0] * VEC_DIM)

		if len(caption) < max_length:
			for j in range(len(caption), max_length):
				caption.append([0] * VEC_DIM)
		
		captions_post.append(caption)

	return captions_post

def caption2list(caption_tuples):
	
	sentences = []

	for i in range(len(caption_tuples)):
		sentence = []
		for word in caption_tuples[i].split():
			sentence.append(word)
		sentences.append(sentence)

	return sentences

def word_to_vec(caption_tuples):

	sentences = caption2list(caption_tuples)
	model = Word2Vec(sentences, min_count=1, size=VEC_DIM)

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

	return max_length




