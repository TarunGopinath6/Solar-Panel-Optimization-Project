import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load data
data = pd.read_csv("output_source.csv")

# Select features and target
features = ["UNIX_TIME", "DC_POWER"]
target = "ANGLE"
X = data[features]
y = data[target]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tfdf.keras.GradientBoostedTreesModel(num_trees=100, max_depth=6)

# Compile the model
model.compile(metrics=["mse", "mae"])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R2 Score:", r2)

# Save the model as a joblib file
joblib.dump(model, "tflite.joblib")

# # Convert the model to a tflite file
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# with open('tflite.tflite', 'wb') as f:
#     f.write(tflite_model)
