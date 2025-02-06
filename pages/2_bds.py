import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
from utils.bds import load_bds_data
from utils.helpers import annotate_bds_listings, create_choropleth_map, create_choropleth_map_avg, create_choropleth_map_simple, load_data, load_geojson, filter_geojson, map_neighborhoods
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

df = load_bds_data(selected_state, selected_city)

neighborhoods_geojson = load_geojson(selected_state, selected_city) if not df.empty else None

# filter neighborhood where feature.properties.Tinh_Thanh = Thành phố Đà Nẵng
neighborhoods_geojson = filter_geojson(neighborhoods_geojson, 'Tinh_Thanh', 'Thành phố Đà Nẵng')

# df with lat, lon
df['coordinates/latitude'] = df['lat'].astype(float)
df['coordinates/longitude'] = df['long'].astype(float)

# drop rows with missing coordinates
df = df.dropna(subset=['coordinates/latitude', 'coordinates/longitude'])

df['neighbourhood_cleansed'] = map_neighborhoods(df, neighborhoods_geojson)

# group by propertyType
propertyType = df.groupby('propertyType').size().reset_index(name='count')
fig = px.pie(propertyType, values='count', names='propertyType', title='Property Type Distribution')
st.plotly_chart(fig)

# select propertyType
propertyTypes = ['House', 'Land', 'Apartment']
selected_propertyType = st.selectbox('Select a property type', propertyTypes)

# select price range
min_price, max_price = df['priceBil'].min(), df['priceBil'].max()
price_range = st.slider('Select a price range', min_value=min_price, max_value=max_price, value=(min_price, max_price))

# select pricePerM2 range
min_pricePerM2, max_pricePerM2 = df['pricePerM2'].min(), df['pricePerM2'].max()
pricePerM2_range = st.slider('Select a price per m2 range', min_value=min_pricePerM2, max_value=max_pricePerM2, value=(min_pricePerM2, max_pricePerM2))

# filter data by selected pricePerM2 range
df = df[(df['pricePerM2'] >= pricePerM2_range[0]) & (df['pricePerM2'] <= pricePerM2_range[1])]

# filter data by selected price range
df = df[(df['priceBil'] >= price_range[0]) & (df['priceBil'] <= price_range[1])]

# filter data by selected propertyType
df = df[df['propertyType'] == selected_propertyType]

# briefly describe the data
st.write(f"Total number of records: {df.shape[0]}")

# total market value
total_market_value = df['priceBil'].sum()
st.write(f"Total market value: {total_market_value} billion VND")


# price distribution
fig = px.histogram(df, x='priceBil', title='Price Distribution')
st.plotly_chart(fig)


choropleth_map_fig = create_choropleth_map_avg(neighborhoods_geojson, df, 'pricePerM2', selected_city, city_centers)

# annotate the map with the data points
choropleth_map_fig = annotate_bds_listings(choropleth_map_fig, df, city=selected_city)
selected_points = plotly_events(choropleth_map_fig, click_event=True, select_event=False, override_height=600, key="neighborhood")


# pricePerM2 avg by neighborhood

pricePerM2_avg = df.groupby('neighbourhood_cleansed')['pricePerM2'].mean().reset_index(name='pricePerM2_avg')

# sort by pricePerM2_avg
pricePerM2_avg = pricePerM2_avg.sort_values('pricePerM2_avg', ascending=False)

fig = px.bar(pricePerM2_avg, x='neighbourhood_cleansed', y='pricePerM2_avg', title='Average Price Per M2 by Neighborhood')
st.plotly_chart(fig)

st.write('Airbnb Data Viz')

# Load data and geojson based on the selections
airbnb_df, cal_df = load_data(selected_state, selected_city)

neighborhoods_geojson = load_geojson(selected_state, selected_city) if not df.empty else None

# filter neighborhood where feature.properties.Tinh_Thanh = Thành phố Đà Nẵng
neighborhoods_geojson = filter_geojson(neighborhoods_geojson, 'Tinh_Thanh', 'Thành phố Đà Nẵng')

airbnb_df['neighbourhood_cleansed'] = map_neighborhoods(airbnb_df, neighborhoods_geojson)

# filter unknown neighbourhood
airbnb_df = airbnb_df[airbnb_df['neighbourhood_cleansed'] != 'Unknown']

# filter price range 0 - 100
airbnb_df = airbnb_df[(airbnb_df['price'] >= 0) & (airbnb_df['price'] <= 100)]

# price_per_person by neighborhood
price_per_person = airbnb_df.groupby('neighbourhood_cleansed')['price_per_person'].mean().reset_index(name='price_per_person_avg')

# sort by price_per_person_avg
price_per_person = price_per_person.sort_values('price_per_person_avg', ascending=False)

fig = px.bar(price_per_person, x='neighbourhood_cleansed', y='price_per_person_avg', title='Average Price Per Person by Neighborhood')
st.plotly_chart(fig)

# ratio of price_per_person to pricePerM2
st.write('Price per person to Price per m2 ratio')
st.write('Higher ratio means higher price per person compared to price per m2 - more profitable for hosts')

price_per_msq = df.groupby('neighbourhood_cleansed')['pricePerM2'].mean().reset_index(name='pricePerM2_avg')
price_per_person = airbnb_df.groupby('neighbourhood_cleansed')['price_per_person'].mean().reset_index(name='price_per_person_avg')

price_ratio = pd.merge(price_per_msq, price_per_person, on='neighbourhood_cleansed')
price_ratio['price_ratio'] = price_ratio['price_per_person_avg'] / price_ratio['pricePerM2_avg']

# sort by price_ratio
price_ratio = price_ratio.sort_values('price_ratio', ascending=False)

fig = px.bar(price_ratio, x='neighbourhood_cleansed', y='price_ratio', title='Price per person to Price per m2 ratio by Neighborhood')
st.plotly_chart(fig)
