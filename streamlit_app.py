
import os
import requests
import streamlit as st
from datetime import date
import pandas as pd

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction | AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background: #0e1728;
    color: #f1f5f9;
    font-size: 14px;
}

/* Sidebar width and background */
[data-testid="stSidebar"] {
    min-width: 270px !important;
    max-width: 270px !important;
    background: #142238;
    border-right: 1px solid #304965;
    font-size: 14px !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h2 {
    font-size: 20px !important;
}

/* Sidebar labels and descriptions */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stCaption {
    font-size: 14px !important;
}

/* Sidebar input text */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] div {
    font-size: 14px !important;
}

/* Main content width */
.block-container {
    max-width: 1350px;
    padding: 2.5rem 2rem;
}

/* Main headings */
.block-container h1 {
    font-size: 2rem !important;
}

.block-container h2 {
    font-size: 1.6rem !important;
}

.block-container h3 {
    font-size: 1.3rem !important;
}

.block-container h4 {
    font-size: 1.1rem !important;
}

/* Main paragraph and text */
.block-container p,
.block-container label,
.block-container li {
    font-size: 14px;
}

/* Hero heading card */
.hero {
    background: linear-gradient(110deg, #183b59, #164b52);
    border: 1px solid #276477;
    border-radius: 22px;
    padding: 26px;
    margin-bottom: 24px;
}

.hero h1 {
    color: #f1f5f9;
    font-size: 2rem !important;
    font-weight: 700;
}

.hero p {
    color: #d5e5f3;
    font-size: 15px;
    margin-bottom: 0;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #192a40;
    border: 1px solid #304965;
    border-radius: 18px;
    padding: 16px;
    min-height: 110px;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    color: #f1f5f9;
    white-space: nowrap;
}

[data-testid="stMetricLabel"] {
    color: #d5e5f3;
    font-size: 14px !important;
}

/* Prediction result card */
.result-box {
    background: #192a40;
    border: 1px solid #365675;
    border-radius: 20px;
    padding: 22px;
    margin-top: 10px;
}

.price-label {
    color: #cbd5e1;
    font-size: 15px;
}

.price-value {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 700;
}

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #304965;
    border-radius: 15px;
    background: #192a40;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    border: 1px solid #328b91;
    background: linear-gradient(90deg, #176477, #168b86);
    color: white;
    font-size: 14px;
    font-weight: 600;
    padding: 10px;
}

.stButton > button:hover {
    border: 1px solid #55c7c0;
    color: white;
}

/* Dividers */
hr {
    border-color: #304965;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# FastAPI Backend URL
# ------------------------------------------------------------
API_URL = os.getenv(
    "API_URL",
    "https://house-price-predict-3a2r.onrender.com/predict"
)

# ------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------
with st.sidebar:

    st.markdown("## 🏡 Property Details")
    st.caption("Enter property details to estimate the house price.")

    location = st.selectbox(
        "Location",
        [
            "Delhi",
            "NCR",
            "Chennai",
            "Pune",
            "Bangalore",
            "Hyderabad",
            "Mumbai"
        ]
    )

    property_type = st.selectbox(
        "Property Type",
        ["Villa", "Apartment", "Penthouse", "Studio"]
    )

    size_sqft = st.number_input(
        "Size (sqft)",
        min_value=100,
        max_value=20000,
        value=1200,
        step=50
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    floor_number = st.number_input(
        "Floor Number",
        min_value=1,
        max_value=100,
        value=2,
        step=1
    )

    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=100,
        value=3,
        step=1
    )

    age_years = st.number_input(
        "Property Age (Years)",
        min_value=0,
        max_value=150,
        value=5,
        step=1
    )

    furnishing = st.selectbox(
        "Furnishing",
        ["Furnished", "Semi-Furnished", "Unfurnished"]
    )

    distance_to_metro_km = st.number_input(
        "Distance to Metro (km)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1
    )

    view_score = st.number_input(
        "View Score",
        min_value=1,
        max_value=10,
        value=7,
        step=1
    )

    date_listed = st.date_input(
        "Date Listed",
        value=date.today()
    )

    predict_clicked = st.button(
        "📊 Predict House Price",
        use_container_width=True
    )

# ------------------------------------------------------------
# Main page
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏡 House Price Prediction</h1>
        <p>
            House price estimation powered by your trained
            Random Forest Regression model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Dashboard metrics
overview1, overview2, overview3 = st.columns(3)

overview1.metric("Model", "Random Forest")
overview2.metric("Input Features", "14")
overview3.metric("Status", "Ready ✓")

st.markdown("### 🏠 Property Price Analysis")

st.write(
    "Adjust the property details in the sidebar and click "
    "**Predict House Price** to estimate its market value."
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if predict_clicked:

    input_data = {
        "location": location,
        "property_type": property_type,
        "size_sqft": int(size_sqft),
        "num_bedrooms": int(bedrooms),
        "num_bathrooms": int(bathrooms),
        "floor_number": int(floor_number),
        "total_floors": int(total_floors),
        "age_years": int(age_years),
        "furnishing": furnishing,
        "distance_to_metro_km": float(distance_to_metro_km),
        "view_score": int(view_score),
        "year": date_listed.year,
        "month": date_listed.month,
        "day": date_listed.day
    }

    try:

        with st.spinner("Analyzing property details..."):

            response = requests.post(
                API_URL,
                json=input_data,
                timeout=180
            )

            response.raise_for_status()
            result = response.json()

        predicted_price = result.get("predicted_price")

        if predicted_price is None:
            st.error("Prediction value not found in API response.")
            st.json(result)
            st.stop()

        predicted_price = float(predicted_price)

        price_crore = predicted_price / 10000000

        st.markdown("### 📊 Prediction Result")

        result_col, details_col = st.columns([1.25, 1])

        with result_col:

            st.markdown(
                f"""
                <div class="result-box">
                    <p class="price-label">
                        Estimated House Price
                    </p>
                    <p class="price-value">
                        ₹ {predicted_price:,.2f}
                    </p>
                    <h3 style="color:#e8f2ff;">
                        Approximately ₹ {price_crore:.2f} Crore
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        with details_col:

            st.markdown("#### 💡 Property Insight")

            st.success(
                f"Estimated property value: ₹ {predicted_price:,.2f}"
            )

            st.metric(
                "Estimated Price per Sqft",
                f"₹ {predicted_price / size_sqft:,.2f}"
            )

        # Property details
        with st.expander("🏡 View Entered Property Details"):

            property_details = {
                "Feature": [
                    "Location",
                    "Property Type",
                    "Size (sqft)",
                    "Bedrooms",
                    "Bathrooms",
                    "Floor Number",
                    "Total Floors",
                    "Property Age",
                    "Furnishing",
                    "Distance to Metro",
                    "View Score",
                    "Date Listed"
                ],

                "Value": [
                    location,
                    property_type,
                    size_sqft,
                    bedrooms,
                    bathrooms,
                    floor_number,
                    total_floors,
                    age_years,
                    furnishing,
                    distance_to_metro_km,
                    view_score,
                    str(date_listed)
                ]
            }

            property_df = pd.DataFrame(property_details)

            st.dataframe(
                property_df.astype(str),
                use_container_width=True,
                hide_index=True
                )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to FastAPI backend. "
            "Please check your Render backend URL."
        )

    except requests.exceptions.Timeout:

        st.error(
            "Backend response timed out. Please try again."
        )

    except requests.exceptions.HTTPError as exc:

        st.error(f"API Error: {exc}")
        st.code(response.text)

    except Exception as exc:

        st.error(f"Prediction failed: {exc}")

else:

    st.info(
        "👈 Enter property details in the sidebar, "
        "then click Predict House Price."
    )

# ------------------------------------------------------------
# Project information
# ------------------------------------------------------------
st.markdown("---")

left, right = st.columns(2)

with left:

    st.markdown("### ℹ️ About this Project")

    st.write(
        "This application uses Machine Learning to predict "
        "residential property prices based on property features "
        "such as location, size, property type, facilities "
        "and other characteristics."
    )

with right:

    st.markdown("### ⚙️ Model Details")

    st.write("**Algorithm:** Random Forest Regressor")

    st.write("**Backend:** FastAPI")

    st.write("**Frontend:** Streamlit")

    st.write("**Deployment:** Render + Streamlit Community Cloud")

st.markdown(
    """
    <p class="small-note">
        Note: This prediction is a Machine Learning estimate.
        Actual market prices may vary.
    </p>
    """,
    unsafe_allow_html=True
)
