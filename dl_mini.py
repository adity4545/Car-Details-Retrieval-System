import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
file_path = 'C:\D DRIVE\dl\Automobile_data.csv'
df = pd.read_csv(file_path)

# Preprocess the dataset
df.columns = df.columns.str.lower()  # Normalize column names
car_name_column = 'make'

# Convert the 'price' column to numeric, handling non-numeric values
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])  # Drop rows with invalid prices
df['price'] = df['price'].astype(float)  # Ensure numeric type

# Helper function for similarity search
def find_closest_match(user_input, car_names):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(car_names)
    input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    closest_idx = similarities.argmax()
    return closest_idx, similarities[closest_idx]

# Streamlit User Interface
st.title("Car Details Retrieval System")
st.write("Search for car details by name or filter by attributes!")

# User Input
search_option = st.radio("Search by:", ("Car Name", "Filters"))

if search_option == "Car Name":
    user_input = st.text_input("Enter the car name:")
    if user_input:
        car_names = df[car_name_column].str.lower().tolist()
        closest_idx, similarity = find_closest_match(user_input.lower(), car_names)
        if similarity > 0.1:  # Threshold for a reasonable match
            st.write("Closest Match Found:", df.iloc[closest_idx][car_name_column])
            st.write("Details:")
            st.dataframe(df.iloc[[closest_idx]])
        else:
            st.write("No close match found. Try refining your input.")
elif search_option == "Filters":
    # Filter options
    fuel_types = df['fuel-type'].dropna().unique().tolist()
    body_styles = df['body-style'].dropna().unique().tolist()
    
    selected_fuel = st.selectbox("Select Fuel Type:", ["All"] + fuel_types)
    selected_body = st.selectbox("Select Body Style:", ["All"] + body_styles)
    price_min = st.number_input("Minimum Price:", min_value=0, value=0)
    price_max = st.number_input("Maximum Price:", min_value=0, value=50000)

    # Apply filters
    filtered_df = df.copy()
    if selected_fuel != "All":
        filtered_df = filtered_df[filtered_df['fuel-type'] == selected_fuel]
    if selected_body != "All":
        filtered_df = filtered_df[filtered_df['body-style'] == selected_body]
    filtered_df = filtered_df[
        (filtered_df['price'] >= price_min) & (filtered_df['price'] <= price_max)
    ]

    if not filtered_df.empty:
        st.write("Filtered Results:")
        st.dataframe(filtered_df)
    else:
        st.write("No cars found matching the selected criteria.")
