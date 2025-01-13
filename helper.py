import tensorflow as tf

model = tf.keras.load_model('model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('tflite_model.tflite', 'wb') as f:
    f.write(tflite_model)