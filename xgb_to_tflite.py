import joblib
import tensorflow as tf
import numpy as np

# Load the saved model
xgb_model = joblib.load('xgb_model.joblib')

# Convert the model to a TensorFlow Lite model
input_arrays = ['UNIX_TIME', 'DC_POWER']
output_arrays = ['ANGLE']
converter = tf.lite.TFLiteConverter.from_sklearn(model=xgb_model, input_arrays=input_arrays, output_arrays=output_arrays)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('xgb_model.tflite', 'wb') as f:
    f.write(tflite_model)
