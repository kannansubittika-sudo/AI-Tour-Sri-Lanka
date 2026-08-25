import pandas as pd
import numpy as np
import requests
import joblib
hotels_df = pd.read_csv("Hotels_Dataset.csv")
restaurants_df = pd.read_csv("Restaurants_Dataset.csv")
from flask import Flask, render_template, request, redirect, session, flash
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
app = Flask(__name__)

app.secret_key = "my_secret_key"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "ai_tour_srilanka"

mysql = MySQL(app)

# ==========================
# Load ML Model & Encoders
# ==========================

model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")

tourism_df = pd.read_csv("SriLanka_Tourism_Dataset.csv")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Planner page
@app.route("/planner")
def planner():
    return render_template("planner.html")

# Login Page
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user[4], password):

            session["user_id"] = user[0]
            session["user_name"] = user[1]
            session["user_email"] = user[2]

            flash("Login Successful!", "success")

            return redirect("/dashboard")

        else:

            flash("Invalid Email or Password!", "danger")

            return redirect("/login")

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        fullname = request.form["fullname"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # Check password match
        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect("/signup")

        # Check if email already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()

        if user:
            flash("Email already exists!", "warning")
            cur.close()
            return redirect("/signup")

        # Hash password
        hashed_password = generate_password_hash(password)

        # Insert user
        cur.execute(
            """
            INSERT INTO users(fullname, email, phone, password)
            VALUES(%s, %s, %s, %s)
            """,
            (fullname, email, phone, hashed_password)
        )

        mysql.connection.commit()
        cur.close()

        flash("Registration Successful! Please Login.", "success")

        return redirect("/login")

    return render_template("signup.html")


# Dashboard Page
@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        flash("Please login first.", "warning")
        return redirect("/login")

    return render_template(
        "dashboard.html",
        username=session["user_name"]
    )


# logout 
@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out successfully.", "success")

    return redirect("/login")


@app.route("/recommend", methods=["POST"])
def recommend():

    # Get form data
    name = request.form["name"]
    district = request.form["district"]
    budget = int(request.form["budget"])
    interest = request.form["interest"]
    budget_level = request.form["budget_level"]
    rating = float(request.form["rating"])

    travelers = request.form["travelers"]
    hotel_type = request.form["hotel_type"]
    transport = request.form["transport"]
    travel_date = request.form["start_date"]
    duration = request.form["days"]

    # Save planner details
    if "user_id" in session:

        cur = mysql.connection.cursor()

        cur.execute("""
            INSERT INTO travel_planner
            (user_id, district, interest, budget, budget_level,
             travelers, hotel_type, transport, rating, travel_date, duration)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            session["user_id"],
            district,
            interest,
            budget,
            budget_level,
            travelers,
            hotel_type,
            transport,
            rating,
            travel_date,
            duration
        ))

        mysql.connection.commit()
        cur.close()

    # Budget Range
    budget_min = budget * 0.8
    budget_max = budget * 1.2

     # Encode Inputs
    district_encoded = encoders["district"].transform([district])[0]
    category_encoded = encoders["category"].transform([interest])[0]
    budget_level_encoded = encoders["budget_level"].transform([budget_level])[0]

    # Model Input
    input_data = np.array([[
        district_encoded,
        category_encoded,
        budget_level_encoded,
        budget_min,
        budget_max,
        rating
    ]])

    # Predict Destination
    prediction = model.predict(input_data)
    predicted_place = encoders["place_name"].inverse_transform(prediction)[0]

    # -----------------------------------
    # Get Place Details
    # -----------------------------------

    place_details = tourism_df[
        (tourism_df["place_name"] == predicted_place) &
        (tourism_df["district"].str.lower() == district.lower())
    ]

    # If model predicts a place from another district,
    # choose the highest rated place from selected district
    if place_details.empty:

        place_details = tourism_df[
            tourism_df["district"].str.lower() == district.lower()
        ]

        place_details = (
            place_details
            .drop_duplicates(subset="place_name")
            .sort_values(by="rating", ascending=False)
            .head(1)
        )

        predicted_place = place_details.iloc[0]["place_name"]

    description = place_details.iloc[0]["description"]
    place_rating = place_details.iloc[0]["rating"]
    latitude = place_details.iloc[0]["latitude"]
    longitude = place_details.iloc[0]["longitude"]

    image_map = {
        "Yala National Wildlife Park": "yala.jpg",
        "Sigiriya": "sigiriya.jpg",
        "Jaffna Clock Tower": "jaffna_clocktower.jpg",
        "Kumana National Park": "kumana_national_wildlife.jpg",
        "Madhu Church": "Madhu_church.jpg",
        "Mannar Fort": "Mannar_fort.jpg",
        "Mirissa": "mirissa.jpg",
        "Nagadeepa Temple": "Nagadeepa_temple.jpg",
        "Nalanda Gedige": "Nalanda_Gediga.jpg",
        "Padaviya": "padaviya.jpg",
        "Panama Beach": "panama_beach.jpg",
        "Sinharaja Forest": "sinharaja_forest.jpg",
        "Thiruketheeswaram Temple": "thiruketheeswaram_temple.jpg",
        "Lankathilaka_temple": "lankathilaka_temple.jpeg",
        "Iranamadu Tank": "iranamadu_tank.jpg",
        "Haputale": "haputale.jpeg",
        "Gangaramaya Temple": "gangaramaya_temple.jpeg"
    }

    image_name = image_map.get(predicted_place, "default.jpg")
    # -----------------------------------
    # Nearby Attractions
    # -----------------------------------
    
    nearby_places = tourism_df[
        (tourism_df["district"].str.lower() == district.lower()) &
        (tourism_df["place_name"] != predicted_place)
    ]

    nearby_places = (
        nearby_places
        .drop_duplicates(subset="place_name")
        .head(4)
        .to_dict("records")
    )
   
    # -----------------------------------
    # Hotels
    # -----------------------------------

    selected_district = district.strip().lower()

    hotels = hotels_df[
        hotels_df["District"].astype(str).str.strip().str.lower()
        == selected_district
    ]

    # Try selected hotel type first
    if hotel_type != "Any":
        filtered_hotels = hotels[
            hotels["Type"].astype(str).str.strip().str.lower()
            == hotel_type.strip().lower()
        ]

        # If selected type is not available,
        # show other hotels from the same district
        if not filtered_hotels.empty:
            hotels = filtered_hotels

    hotels = hotels.head(3).to_dict("records")


    # -----------------------------------
    # Restaurants
    # -----------------------------------

    restaurants = restaurants_df[
        restaurants_df["District"].astype(str).str.strip().str.lower()
        == selected_district
    ]

    restaurants = restaurants.head(3).to_dict("records")
    return render_template(
        "recommendation.html",
        name=name,
        place=predicted_place,
        district=district,
        budget=budget,
        interest=interest,
        rating=place_rating,
        description=description,
        travelers=travelers,
        duration=duration,
        hotel_type=hotel_type,
        transport=transport,
        travel_date=travel_date,
        hotels=hotels,
        restaurants=restaurants,
        nearby_places=nearby_places,
        latitude=latitude,
        longitude=longitude,
        image_name=image_name
    )

@app.route("/restaurants")
def restaurant_page():

    restaurants = restaurants_df.head(50).to_dict("records")

    return render_template(
        "restaurants.html",
        restaurants=restaurants
    )

@app.route("/hotels")
def hotel_page():
    hotels = hotels_df.to_dict("records")
    return render_template("hotels.html", hotels=hotels)

@app.route("/weather", methods=["GET", "POST"])
def weather():

    weather_data = None

    if request.method == "POST":

        city = request.form.get("city")

        try:

            # Find district location
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"

            geo_params = {
                "name": city,
                "count": 1,
                "countryCode": "LK",
                "language": "en",
                "format": "json"
            }

            geo_response = requests.get(
                geo_url,
                params=geo_params,
                timeout=10
            )

            geo_data = geo_response.json()

            if geo_data.get("results"):

                latitude = geo_data["results"][0]["latitude"]
                longitude = geo_data["results"][0]["longitude"]

                # Get current weather
                weather_url = "https://api.open-meteo.com/v1/forecast"

                weather_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "timezone": "Asia/Colombo"
                }

                weather_response = requests.get(
                    weather_url,
                    params=weather_params,
                    timeout=10
                )

                data = weather_response.json()

                current = data["current"]

                weather_code = current["weather_code"]

                # Weather condition
                if weather_code == 0:
                    condition = "Clear Sky"
                elif weather_code in [1, 2, 3]:
                    condition = "Partly Cloudy"
                elif weather_code in [45, 48]:
                    condition = "Foggy"
                elif weather_code in [51, 53, 55, 56, 57]:
                    condition = "Drizzle"
                elif weather_code in [61, 63, 65, 66, 67]:
                    condition = "Rainy"
                elif weather_code in [71, 73, 75, 77]:
                    condition = "Snow"
                elif weather_code in [80, 81, 82]:
                    condition = "Rain Showers"
                elif weather_code in [95, 96, 99]:
                    condition = "Thunderstorm"
                else:
                    condition = "Unknown"

                weather_data = {
                    "city": city,
                    "temperature": current["temperature_2m"],
                    "humidity": current["relative_humidity_2m"],
                    "wind": current["wind_speed_10m"],
                    "condition": condition
                }

            else:

                weather_data = {
                    "city": city,
                    "temperature": "N/A",
                    "humidity": "N/A",
                    "wind": "N/A",
                    "condition": "Weather data not available"
                }

        except Exception as e:

            print("Weather Error:", e)

            weather_data = {
                "city": city,
                "temperature": "N/A",
                "humidity": "N/A",
                "wind": "N/A",
                "condition": "Unable to get weather data"
            }

    return render_template(
        "weather.html",
        weather=weather_data
    )

if __name__ == "__main__":
    app.run(debug=True)