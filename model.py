import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Crop_recommendation.csv")

print(df.head(5))
print(df.shape)
print(df.isnull().sum())

x = df.iloc[:,:-1] # select all rows and columns except the last one
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = model.score(x_test, y_test)

print(accuracy)

pickle.dump(model, open("model.pkl", "wb"))

print("Done")