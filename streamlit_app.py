
import streamlit as st
import requests
from datetime import date

st.title("House Price Prediction")

## input widgets

location = st.selectbox(
    "Location",
    ['Delhi','NCR','Chennai', 'Pune', 'Bangalore',
     'Hyderabad', 'Mumbai']
)

property_type = st.selectbox(
    "Property_type",
    ['Villa', 'Apartment', 'Penthouse', 'Studio']
)

size_sqft = st.number_input("Size_sqft", 351, 4997,1783 )

num_bedrooms = st.number_input("Num_bedrooms", 1,5,5 )

num_bathrooms = st.number_input("Num_bathrooms ", 1,4,2)

floor_number = st.number_input("Floor_number ", 1,29,6)

total_floors = st.number_input("Total_floors ", 5,39,25)

age_years = st.number_input("Age_years ", 0,29,4)

furnishing = st.selectbox(
    "Furnishing",
    ['Fully Furnished','Semi-Furnished','Unfurnished']
)

distance_to_metro_km = st.number_input("Distance_to_metro_km ", 0.11,10.0,1.75)

view_score = st.number_input("View_score ", 1,5,5)

# Automatically generate date features
today = date.today()

year = today.year
month = today.month
day = today.day


# Prediction button
if st.button("Predict House Price"):

    data = {
        "location": location,
        "property_type": property_type,
        "size_sqft": size_sqft,
        "num_bedrooms": num_bedrooms,
        "num_bathrooms": num_bathrooms,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "age_years": age_years,
        "furnishing": furnishing,
        "distance_to_metro_km": distance_to_metro_km,
        "view_score": view_score,
        "year": year,
        "month": month,
        "day": day
    }


    
    
    try:
        with st.spinner("Predicting house price..."):

            response = requests.post(
                "https://house-price-predict-3a2r.onrender.com/predict",
                json=data,
                timeout=180
            )

            response.raise_for_status()

            result = response.json()

            st.success(result["result"])

    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")

    except requests.exceptions.HTTPError:
        st.error(f"API Error: {response.status_code}")
        st.code(response.text)

    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
