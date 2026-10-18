from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

df = pd.DataFrame(
    {
        "area": [900, 1200, 1600, 2000, 2400],
        "bedrooms": [2, 3, 3, 4, 4],
        "price": [35, 48, 65, 82, 98],
    }
)
X = df[["area", "bedrooms"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
