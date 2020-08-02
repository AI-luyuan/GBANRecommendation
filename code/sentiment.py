import numpy as np
import random
#import h5py
import os
import time
import collections
from embeddings import Embeddings 
from scipy import sparse
class Sentiment(object):
	"""NP_chunking data preparation"""
	def __init__(self, data_name, num_class=5):
		self.data_name = data_name
		self.train_data_path = '../../data/' + self.data_name + '/demo_train.txt'
		self.test_data_path = '../../data/' + self.data_name + '/demo_test.txt'
		# self.train_data_path = '../data/' + self.data_name + '/train.txt'
		# self.test_data_path = '../data/' + self.data_name + '/test.txt'
		self.train_one_hot_path = '../../data/' + self.data_name + '/train_one_hot.npz'
		self.test_one_hot_path = '../../data/' + self.data_name + '/test_one_hot.npz'
		self.embeddings = Embeddings(data_name)
		self.num_class = num_class
		start_time = time.time()
		self.load_data()
		print ('Reading datasets comsumes %.3f seconds' % (time.time()-start_time))
			
	def deal_with_data(self, path):
		users, products, labels, docs, len_docs, len_words, pdocs = [], [], [], [], [], [], []
		k = 0
		for line in open(path, 'r', encoding='UTF-8'):
			tokens = line.strip().split('\t\t')
			users.append(tokens[0])
			products.append(tokens[1])
			labels.append(int(tokens[2]))
			doc = tokens[3].strip().split('<sssss>')
			# len_docs.append(len(doc))
			doc = [sentence.strip().split(' ') for sentence in doc]
			# len_words.append([len(sentence) for sentence in doc])
			docs.append(doc)
			pdoc = tokens[4].strip().split(' ')
			pdocs.append(pdoc)
		

			k += 1
		# print(pdocs)
		return users, products, labels, docs, pdocs

	def load_data(self):
		print('train_load')
		train_users, train_products, train_labels, train_docs, train_pdocs = self.deal_with_data(self.train_data_path)
		print('test_load')
		test_users, test_products, test_labels, test_docs, test_pdocs = self.deal_with_data(self.test_data_path)
		print('2id')
		train_one_hot = sparse.load_npz(self.train_one_hot_path)
		test_one_hot = sparse.load_npz(self.test_one_hot_path)


		train_docs = self.embeddings.docs2ids(train_docs)
		# print(train_docs)
		test_docs = self.embeddings.docs2ids(test_docs)

		train_pdocs = self.embeddings.cand2ids(train_pdocs)
		# print(train_pdocs)
		test_pdocs = self.embeddings.cand2ids(test_pdocs)

		train_users = self.embeddings.users2ids(train_users)
		test_users = self.embeddings.users2ids(test_users)

		train_products = self.embeddings.prdts2ids(train_products)
		test_products = self.embeddings.prdts2ids(test_products)
		self.train_one_hot = train_one_hot
		self.test_one_hot = test_one_hot

		self.train_set = list(zip(train_docs, train_labels, train_users, train_products,train_pdocs))
		self.test_set = list(zip(test_docs, test_labels, test_users, test_products,test_pdocs))

