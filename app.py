 
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

 

model = joblib.load("model.pkl")
encoder = joblib.load("city_encoder.pkl")
 

data = pd.read_csv("data/india_2000_2024_daily_weather.csv")

cities = sorted(data["city"].unique())


@app.route("/")
def home():

    return render_template(
        "index.html",
        cities=cities,
        prediction_text=""
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        

        city_name = request.form["city"]
        date = request.form["date"]

         

        row = data[
            (data["city"] == city_name) &
            (data["date"] == date)
        ]

        if row.empty:

            return render_template(
                "index.html",
                cities=cities,
                prediction_text="No data found for selected City and Date."
            )
 

        city = encoder.transform([city_name])[0]

        
        date_obj = pd.to_datetime(date)

        date_value = (
         date_obj.year * 10000 +
         date_obj.month * 100 +
         date_obj.day
       )
 
        temperature_2m_min = row["temperature_2m_min"].iloc[0]

        apparent_temperature_max = row["apparent_temperature_max"].iloc[0]

        apparent_temperature_min = row["apparent_temperature_min"].iloc[0]

        precipitation_sum = row["precipitation_sum"].iloc[0]

        rain_sum = row["rain_sum"].iloc[0]

        weather_code = row["weather_code"].iloc[0]

        wind_speed_10m_max = row["wind_speed_10m_max"].iloc[0]

        wind_gusts_10m_max = row["wind_gusts_10m_max"].iloc[0]

        wind_direction_10m_dominant = row["wind_direction_10m_dominant"].iloc[0]
         

        features = [[

            city,

            date_value,

            temperature_2m_min,

            apparent_temperature_max,

            apparent_temperature_min,

            precipitation_sum,

            rain_sum,

            weather_code,

            wind_speed_10m_max,

            wind_gusts_10m_max,

            wind_direction_10m_dominant

        ]]

        prediction = model.predict(features)[0]

        
        return render_template(

            "index.html",

            cities=cities,

            prediction_text=f"Predicted Maximum Temperature : {prediction:.2f} °C"

        )

    except Exception as e:

        return render_template(

            "index.html",

            cities=cities,

            prediction_text=f"Error : {e}"

        )


if __name__ == "__main__":

    app.run(debug=True)