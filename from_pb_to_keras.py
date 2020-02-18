import tensorflow as tf
import numpy as np
from keras_applications.mobilenet_v3 import MobileNetV3
# pretrained tf graphs available at https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
# mobilenet_v3.py you should put to
# C:\Users\...\AppData\Local\Programs\Python\Python36\Lib\site-packages\keras_applications\

# 'large' or 'small'
MODEL_TYPE = 'large'
# 1.0 or 0.75
ALPHA = 0.75
# '-minimalistic' or ''
MINIMALISTIC = ''
INCLUDE_TOP = True
NAME = 'v3-' + MODEL_TYPE + MINIMALISTIC + '_224_' + str(ALPHA) + '_float'
PATH_TO_PB_FILE = NAME + '/' + NAME + '.pb'
SAVE_PATH = ''

# loading weights from pb file
with tf.compat.v1.Session() as sess:
    constant_values = []
    with tf.io.gfile.GFile(PATH_TO_PB_FILE, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        constant_ops = [op for op in sess.graph.get_operations() if op.type == 'Const']
        for constant_op in constant_ops:
            temp = sess.run(constant_op.outputs[0])
            if temp.shape == () or temp.shape == (1,):  # removing relu, add, mull, etc. layers
                continue
            constant_values.append(sess.run(constant_op.outputs[0]))

    values = constant_values
    if INCLUDE_TOP:
        values = values[:-1]
        values[-2] = np.delete(values[-2], 0, axis=-1)
        values[-1] = np.delete(values[-1], 0, axis=-1)
    else:
        values = values[:-3]

if MINIMALISTIC == '':
    minim = False
else:
    minim = True
model = MobileNetV3(input_shape=(224, 224, 3), include_top=INCLUDE_TOP, alpha=ALPHA, weights=None, model_type=MODEL_TYPE, minimalistic=minim,
                    regularizer=tf.keras.regularizers.l2(1e-5),
                    layers=tf.keras.layers, backend=tf.keras.backend, models=tf.keras.models, utils=tf.keras.utils)
model.summary()
model.set_weights(values)
model.save(SAVE_PATH + NAME + '.h5')
model.save_weights(SAVE_PATH + 'weights_' + NAME + '.h5')
