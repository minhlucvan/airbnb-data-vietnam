import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.geo import calc_distance_to_coast, calc_distance_to_point, find_place_by_name
from utils.helpers import annotate_airbnb_listings, create_choropleth_map_simple, filter_geojson, load_data, load_geojson, load_geojson_from_path, map_neighborhoods
from streamlit_plotly_events import plotly_events
st.set_page_config(page_title="Airbnb Data Viz", page_icon='dashboard/  airbnb_l.svg', layout='wide', initial_sidebar_state='auto')

# st.set_page_config(layout="wide")

states = ['Vietnam']

cities = {
    'Vietnam': ['Danang']
}
city_centers = {
    'Danang': {'lat': 16.0544, 'lon': 108.2022}
}

# Sidebar for state and city selection
selected_state = st.sidebar.selectbox('Select a state', states, key='state')
selected_city = st.sidebar.selectbox('Select a city', cities[selected_state], key='city')

# Load data and geojson based on the selections
df, cal_df = load_data(selected_state, selected_city)

neighborhoods_geojson = load_geojson(selected_state, selected_city) if not df.empty else None
coast_geojson = load_geojson_from_path('data/vietnam/danang/coast.geojson') if not df.empty else None
tourism_geojson = load_geojson_from_path('data/vietnam/danang/tourism.geojson') if not df.empty else None

# filter neighborhood where feature.properties.Tinh_Thanh = Thành phố Đà Nẵng
neighborhoods_geojson = filter_geojson(neighborhoods_geojson, 'Tinh_Thanh', 'Thành phố Đà Nẵng')

df['neighbourhood_cleansed'] = map_neighborhoods(df, neighborhoods_geojson)
df['distance_to_coast'] = calc_distance_to_coast(df, coast_geojson)

# filter only entire home/apt
# df = df[df['room_type'] == 'Entire home/apt']

# filter unknown neighbourhood
df = df[df['neighbourhood_cleansed'] != 'Unknown']

# filter price range 0 - 100
df = df[(df['price'] >= 0) & (df['price'] <= 100)]

all_distances = []

# loop through tourism places, calculate distance to each place
for feature in tourism_geojson['features']:
    name = feature['properties']['name']
    name_code = name.lower().replace(' ', '_')
    # utf-8 to ascii
    name_code = name_code.encode('ascii', 'ignore').decode('ascii')
    
    # get feature index, find index of feature in tourism_geojson
    feature_index = tourism_geojson['features'].index(feature)
    
    name_code = name_code if name_code != 'unnamed' else feature_index
    
    lon, lat = feature['geometry']['coordinates']
    all_distances.append(f'distance_to_{name_code}')
    df[f'distance_to_{name_code}'] = calc_distance_to_point(df, lon, lat)


features_df = df[['name', 'price', 'accommodates', 'neighbourhood_cleansed', 'distance_to_coast'] + all_distances]


# Add these imports at the top with other imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ... [Keep all your existing code until the features_df line] ...

# Prepare data for modeling
X = df[['neighbourhood_cleansed', 'distance_to_coast', 'accommodates'] + all_distances]
y = df['price']

st.write("Data for modeling")

# Define preprocessing pipeline
categorical_features = ['neighbourhood_cleansed']
numerical_features = ['distance_to_coast', 'accommodates'] + all_distances

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Initialize and train model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
st.header("Model Performance")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

st.write(f"**Root Mean Squared Error:** ${rmse:.2f}")
st.write(f"**Mean Absolute Error:** ${mae:.2f}")

# Feature importance analysis
feature_names = (
    model.named_steps['preprocessor']
    .named_transformers_['cat']
    .get_feature_names_out(categorical_features)
).tolist() + numerical_features

importances = model.named_steps['regressor'].feature_importances_

st.subheader("Top 10 Important Features")
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
st.bar_chart(importance_df.set_index('Feature'))
