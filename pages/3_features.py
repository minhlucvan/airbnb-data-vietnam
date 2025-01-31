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


# hexagon_geojson = load_geojson_from_path('data/vietnam/danang/boudary-h3.geojson') if not df.empty else None
choropleth_map_fig = create_choropleth_map_simple(neighborhoods_geojson, df, selected_city, city_centers)
choropleth_map_fig = annotate_airbnb_listings(choropleth_map_fig, df, selected_city)
selected_points = plotly_events(choropleth_map_fig, click_event=True, select_event=False, override_height=600, key="choropleth_map")

# distance to coast
fig = px.scatter(df, x='distance_to_coast', y='price_per_person',
                color='price_per_person', hover_data=['name'],
                trendline='lowess', trendline_color_override='red',
                labels={'distance_to_coast': 'Distance to coast (km)', 'price_per_person': 'Price per person (USD)'})
st.plotly_chart(fig, key='distance_to_coast')

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
    
st.write('Distance to tourism places')

sample_places = ['Airport', "Dragon's Head (Fire Show)", ]
for place in sample_places:
    feature = find_place_by_name(tourism_geojson, place)
    name = feature['properties']['name']
    name_code = name.lower().replace(' ', '_')
    # utf-8 to ascii
    name_code = name_code.encode('ascii', 'ignore').decode('ascii')
    
    feature_index = tourism_geojson['features'].index(feature)
    
    name_code = name_code if name_code != 'unnamed' else  feature_index
    
    
    fig = px.scatter(df, x=f'distance_to_{name_code}', y='price_per_person',
                    color='price_per_person', hover_data=['name'],
                    trendline='lowess', trendline_color_override='red',
                    labels={f'distance_to_{name_code}': f'Distance to {name} (km)', 'price_per_person': 'Price per person (USD)'})
    st.plotly_chart(fig, key=f'distance_to_{name_code}')


features_df = df[['name', 'price_per_person', 'neighbourhood_cleansed', 'distance_to_coast'] + all_distances]
