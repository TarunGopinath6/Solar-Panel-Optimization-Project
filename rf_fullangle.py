import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from micromlgen import port
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus


# Load dataset
df = pd.read_csv("output_source.csv")

# Split into features and target
X = df[["UNIX_TIME", "DC_POWER"]]
y = df["ANGLE_FULL"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

#Printing the tree
# Extract a single tree from the forest
tree = rf_model.estimators_[0]

# Export the tree to a Graphviz dot file
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=X.columns,  
                           filled=True, rounded=True,  
                           special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png("tree.png")


# Predict on test set
y_pred = rf_model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(rf_model, 'rf_model_full.joblib')

outs = pd.DataFrame()
outs['UNIX_TIME'] = pd.Series(X_test['UNIX_TIME'].values)
outs['ANGLE_FULL'] = pd.Series(y_pred)
outs.sort_values(by="UNIX_TIME", ascending=True)
outs.to_csv('rf_model_full.csv', encoding='utf-8', index=False)

print("MSE:", mse)
print("R-squared:", r2)
with open('rf_model_full.c', 'w') as f:
    f.write(port(rf_model))
