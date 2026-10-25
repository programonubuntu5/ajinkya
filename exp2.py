
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Actual:", y_test)
print("Predicted:", pred)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

print("Prediction for 11:", model.predict([[11]]))
