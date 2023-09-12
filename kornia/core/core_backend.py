import keras_core as keras

# classes

Module = keras.layers.Layer
ModuleList = keras.Sequential

# functions
# NOTE: ideally we expose what we find in numpy
arange = keras.ops.arange
concatenate = keras.ops.concatenate
stack = keras.ops.stack
linspace = keras.ops.linspace
normalize = keras.utils.normalize
pad = keras.ops.pad
eye = keras.ops.eye
einsum = keras.ops.einsum
zeros = keras.ops.zeros
zeros_like = keras.ops.zeros_like
ones = keras.ops.ones
ones_like = keras.ops.ones_like
where = keras.ops.where
diag = keras.ops.diag
softmax = keras.ops.softmax

# random
rand = keras.random
