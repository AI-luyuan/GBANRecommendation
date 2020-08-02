import numpy as np
import random
#import h5py
import os
import time
import collections

import pickle as pk 
import os 

class Embeddings(object):
	def __init__(self, data_name, dim=200):
		self.data_name = data_name
		self.word_file = '../../data/' + self.data_name + '/word.txt'
		self.word_emb_file = '../../data/' + self.data_name + '/embword.save'
		# self.user_file = '../data/' + self.data_name + '/usrlist.txt'
		# self.user_emb_file = '../data/' + self.data_name + '/embuser.save'
		# self.prdt_file = '../data/' + self.data_name + '/prdlist.txt'
		# self.prdt_emb_file = '../data/' + self.data_name + '/embprdt.save'
		self.dim = dim

		# if os.path.isfile(self.user_emb_file) == False:
		# 	self.build_user_embedding()
		# if os.path.isfile(self.prdt_emb_file) == False:
		# 	self.build_prdt_embedding()

		self.word2id, self.word_embeddings = self.load_word_embedding()
		# self.user2id, self.user_embeddings = self.load_user_embedding()
		# self.prdt2id, self.prdt_embeddings = self.load_prdt_embedding()


	def load_word_embedding(self):
		words = ['UNK']
		for line in open(self.word_file, 'r', encoding='utf-8'):
			word = line.strip()
			words.append(word)
		
		word_embeddings = pk.load(open(self.word_emb_file, 'rb'))
		# print (len(words))
		# print (word_embeddings.shape)
		unk_embedding = np.random.rand(self.dim)*2-1
		word_embeddings = np.insert(word_embeddings, 0, values=unk_embedding, axis=0)

		word2id = dict(zip(words, range(len(words))))
		return word2id, word_embeddings



	def docs2ids(self, docs):
		id_set = []
		for doc in docs:
			id_set.append([])
			for sentence in doc:
				id_set[-1].append([])
				for word in sentence:
					if word not in self.word2id:
						id_set[-1][-1].append(self.word2id['UNK'])
					else:
						id_set[-1][-1].append(self.word2id[word])
		return id_set


	def cand2ids(self, doc):
		id_set = []	
		for sentence in doc:
			id_set.append([])
			for word in sentence:
				if word not in self.word2id:
					id_set[-1].append(self.word2id['UNK'])
				else:
					id_set[-1].append(self.word2id[word])
		return id_set



	def users2ids(self, users):
		id_set = []
		for user in users:
			if user not in self.word2id:
				id_set.append(self.word2id['UNK'])
			else:
				id_set.append(self.word2id[user])
		return id_set

	def prdts2ids(self, prdts):
		id_set = []
		for prdt in prdts:
			if prdt not in self.word2id:
				id_set.append(self.word2id['UNK'])
			else:
				id_set.append(self.word2id[prdt])
		return id_set

