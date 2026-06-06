import tensorflow as tf

model = tf.keras.layers.TFSMLayer(
    "Keras/model.savedmodel",
    call_endpoint="serving_default"
)

print("SavedModel berhasil dibaca")