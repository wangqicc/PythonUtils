import os
import tensorflow as tf
from tensorflow import keras

import numpy as np
np.set_printoptions(threshold=np.inf)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print(len(train_data[0]), len(train_data[1]))

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(train_data[0])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

# 输入形状是用于电影评论的词汇数目（10,000 词）
EPOCH = 5
BATCH_SIZE = 512
base_path = '../data'
vocab_size = 10000
MAX_LEN = 256

model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=vocab_size, output_dim=16, input_length=MAX_LEN))
model.add(keras.layers.Conv1D(filters=12, kernel_size=5, strides=1, padding='valid'))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

model.load_weights(filepath=os.path.join(base_path, 'model1/cp-0001.ckpt'))

model.get_layer().get_weights()

print('input: {}, input shape: {}'.format(x_val[0], x_val[0].shape))

# embedding_layer = keras.Model(inputs=model.input, outputs=model.get_layer('embedding').output)
# embedding = embedding_layer.predict(x_val[0])
# print('embedding: {}'.format(embedding))
# for item in embedding:
#     for num in item[0]:
#         print(num, end=' ')
#     print()

conv1d_layer = keras.Model(inputs=model.input, outputs=model.get_layer('conv1d').output)
print('conv1d: {}'.format(conv1d_layer.predict(tf.expand_dims(x_val[0], 0))))
