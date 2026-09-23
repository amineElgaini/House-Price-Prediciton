import streamlit as st
import pandas as pd
import joblib

# Load once, cached so it doesn't reload on every interaction
@st.cache_resource
def load_assets():
    model = joblib.load("../models/ridge_model.pkl")
    template = joblib.load("../models/template_row.pkl")
    neighborhoods = joblib.load("../models/neighborhoods.pkl")
    return model, template, neighborhoods

model, template, neighborhoods = load_assets()

st.title("House Price Estimator")
st.write("Enter the main characteristics of a house to estimate its sale price.")

# --- Form ---
col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 6)
    total_bsmt_sf = st.number_input("Basement area (sq ft)", 0, 3000, 800)
    garage_cars = st.slider("Garage capacity (cars)", 0, 4, 2)
    total_bath = st.slider("Total bathrooms", 1.0, 5.0, 2.0, step=0.5)

with col2:
    year_built = st.number_input("Year built", 1870, 2010, 1990)
    neighborhood = st.selectbox("Neighborhood", neighborhoods)
    house_age = 2010 - year_built
    first_flr_sf = st.number_input("1st floor area (sq ft)", 300, 4000, 1000)
    second_flr_sf = st.number_input("2nd floor area (sq ft)", 0, 2500, 0)

gr_liv_area = first_flr_sf + second_flr_sf 

# --- Build the full input row ---
row = template.copy()
row["OverallQual"] = overall_qual
row["GrLivArea"] = gr_liv_area
row["TotalBsmtSF"] = total_bsmt_sf
row["GarageCars"] = garage_cars
row["TotalBath"] = total_bath
row["YearBuilt"] = year_built
row["HouseAge"] = house_age
row["Neighborhood"] = neighborhood
row["1stFlrSF"] = first_flr_sf
row["2ndFlrSF"] = second_flr_sf
row["TotalSF"] = total_bsmt_sf + first_flr_sf + second_flr_sf

input_df = pd.DataFrame([row])

# # --- Predict ---
# if st.button("Estimate price"):
#     prediction = model.predict(input_df)[0]
#     st.success(f"Estimated sale price: ${prediction:,.0f}")

#     st.subheader("Where this house stands")
#     st.write(f"- Living area: **{gr_liv_area} sq ft** (dataset median: {int(template['GrLivArea'])} sq ft)")
#     st.write(f"- Overall quality: **{overall_qual}/10**")
#     st.write(f"- Neighborhood: **{neighborhood}**")






import matplotlib.pyplot as plt

@st.cache_resource
def load_prices():
    return joblib.load("../models/y_train.pkl")

if st.button("Estimate price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated sale price: ${prediction:,.0f}")

    y_train_prices = load_prices()
    percentile = (y_train_prices < prediction).mean() * 100
    st.write(f"This estimate is higher than **{percentile:.0f}%** of houses in the training data.")

    fig, ax = plt.subplots()
    ax.hist(y_train_prices, bins=40, color="lightgray")
    ax.axvline(prediction, color="red", linestyle="--", label="Your estimate")
    ax.set_xlabel("Sale Price")
    ax.set_title("Where this estimate falls in the market")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Where this house stands")
    st.write(f"- Living area: **{gr_liv_area} sq ft** (dataset median: {int(template['GrLivArea'])} sq ft)")
    st.write(f"- Overall quality: **{overall_qual}/10**")
    st.write(f"- Neighborhood: **{neighborhood}**")