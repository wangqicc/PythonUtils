import os
import numpy as np
from tensorflow import keras


# save 3d array
def save_3d_array(fname, X, fmt):
    with open(fname, 'w', encoding='utf-8') as f:
        f.write('{} {} {} '.format(X.shape[0], X.shape[1], X.shape[2]))
        for idx in range(X.shape[0]):
            f.write('# #\n')
            np.savetxt(fname=f, X=X[idx], fmt=fmt)
        f.close()


# save weight
def save_weight(base_path, model):
    model.load_weights(filepath=os.path.join(base_path, 'cp-0001.ckpt'))
    for idx in range(len(model.layers)):
        layer = model.get_layer(index=idx)
        layer_name = str(layer.name)

        if 'embedding' == layer_name:
            weights = layer.get_weights()
            np.savetxt(fname=os.path.join(base_path, 'embedding'), X=weights[0], fmt='%.18e')

        if 'conv1d' == layer_name:
            weights = layer.get_weights()
            save_3d_array(fname=os.path.join(base_path, 'conv1d_weight'), X=weights[0], fmt='%.18e')
            np.savetxt(fname=os.path.join(base_path, 'conv1d_bias'), X=weights[1], fmt='%.18e')

        if 'dense' == layer_name:
            weights = layer.get_weights()
            np.savetxt(fname=os.path.join(base_path, 'dense_weight'), X=weights[0], fmt='%.18e')
            np.savetxt(fname=os.path.join(base_path, 'dense_bias'), X=weights[1], fmt='%.18e')

        if 'dense_1' == layer_name:
            weights = layer.get_weights()
            np.savetxt(fname=os.path.join(base_path, 'dense_1_weight'), X=weights[0], fmt='%.18e')
            np.savetxt(fname=os.path.join(base_path, 'dense_1_bias'), X=weights[1], fmt='%.18e')


if __name__ == '__main__':
    MAX_LEN = 256
    VOCAB_SIZE = 10000
    base_path = '../data/model1'

    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=16, input_length=MAX_LEN))
    model.add(keras.layers.Conv1D(filters=12, kernel_size=5, strides=1, padding='valid'))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    save_weight(base_path, model)
    pass
