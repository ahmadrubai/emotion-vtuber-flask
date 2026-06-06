import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Physical GPU devices:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")

if not gpus:
    print("[INFO] GPU tidak terdeteksi oleh TensorFlow.")
    print("[INFO] Training akan berjalan di CPU.")
else:
    print("[INFO] GPU terdeteksi.")
    for gpu in gpus:
        print(gpu)