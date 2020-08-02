from sentiment import Sentiment
from model_rec import Model
import tensorflow as tf

tf.app.flags.DEFINE_string("task", "np_chunking", "Task.")
tf.app.flags.DEFINE_string("cell", "lstm", "Rnn cell.")
tf.app.flags.DEFINE_integer("size", 50, "Size of each layer.")
tf.app.flags.DEFINE_integer("batch", 1000, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 100, "Number of training epoch.")
tf.app.flags.DEFINE_string("loss", "cross_entropy", "Loss function.")
tf.app.flags.DEFINE_float("dropout", 0.9, "Dropout keep probability.")
tf.app.flags.DEFINE_string("gpu", '0', "GPU id.")
tf.app.flags.DEFINE_string("cpu", '0', "CPU id")
tf.app.flags.DEFINE_string("opt",'Adagrad','Optimizer.')

FLAGS = tf.app.flags.FLAGS

data_name = 'movies'

def train():
	print("load data")
	d = Sentiment(data_name, 6)
	print("start_train")
	with tf.device('/gpu:'+FLAGS.gpu):
		m = Model(data_name, d.num_class, embeddings = d.embeddings, size = FLAGS.size, 
			batch_size = FLAGS.batch, dropout = FLAGS.dropout,
			rnn_cell = FLAGS.cell, optimize = FLAGS.opt)
		print("start_fit")
		m.fit(d.train_set, d.test_set, d.train_one_hot, d.test_one_hot, FLAGS.epoch)

if __name__ == '__main__':
	train()
