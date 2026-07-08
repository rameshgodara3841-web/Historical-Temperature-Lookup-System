 
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

 

data = pd.read_csv("data/india_2000_2024_daily_weather.csv")

data.dropna(inplace=True)

print(data.head())
print(data.columns)

 

encoder = LabelEncoder()

data["city"] = encoder.fit_transform(data["city"])

joblib.dump(encoder, "city_encoder.pkl")
 

data["date"] = pd.to_datetime(data["date"])

 
data["date"] = (
    data["date"].dt.year * 10000 +
    data["date"].dt.month * 100 +
    data["date"].dt.day
)
  

X = data[[
    "city",
    "date",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "weather_code",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant"
]]

y = data["temperature_2m_max"]

 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Train Shape :", X_train.shape)
print("Test Shape :", X_test.shape)
 

model = LinearRegression()

model.fit(X_train, y_train)
 
prediction = model.predict(X_test)

score = r2_score(y_test, prediction)

print("\nR2 Score :", score)

 

joblib.dump(model, "model.pkl")

print("\nModel Saved Successfully")
