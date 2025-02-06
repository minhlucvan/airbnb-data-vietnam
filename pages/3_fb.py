import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
from utils.bds import load_bds_data
from utils.fb import load_fb_data
from utils.gpt import process_csv
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

df = load_fb_data(selected_state, selected_city)

neighborhoods_geojson = load_geojson(selected_state, selected_city) if not df.empty else None

# filter neighborhood where feature.properties.Tinh_Thanh = Thành phố Đà Nẵng
neighborhoods_geojson = filter_geojson(neighborhoods_geojson, 'Tinh_Thanh', 'Thành phố Đà Nẵng')

# list features names
features = neighborhoods_geojson['features']
feature_names = [feature['properties']['Name'] for feature in features]


districts = neighborhoods_geojson['features']
district_names = [district['properties']['District'] for district in districts]
unique_district_names = list(set(district_names))

# where city = Danang
df = df[df['city'] == 'Danang']

# where propertyType = land or house
df = df[df['propertyType'].isin(['land', 'house'])]

# drop non number value
df = df[pd.to_numeric(df['priceBil'], errors='coerce').notnull()]

df['priceBil'] = df['priceBil'].astype(float)

# price range
price_slider = st.slider('Select price range', min_value=float(df['priceBil'].min()), max_value=float(df['priceBil'].max()), value=(float(df['priceBil'].min()), float(df['priceBil'].max())))

# filter price
df = df[(df['priceBil'] >= price_slider[0]) & (df['priceBil'] <= price_slider[1])]

# convert area to float
df['area'] = pd.to_numeric(df['area'], errors='coerce')
df = df[df['area'].notnull()]
df['area'] = df['area'].astype(float)

st.dataframe(df)

# post by day
post_by_day = df.groupby('date').size().reset_index(name='count')

fig = px.line(post_by_day, x='date', y='count', title='Post by Day')
st.plotly_chart(fig)

# Normalize district names
district_mapping = {
    'Hai Chau District': 'Hai Chau District',
    'Ngu Hanh Son District': 'Ngu Hanh Son District',
    'Lien Chieu District': 'Lien Chieu District',
    'Son Tra District': 'Son Tra District',
    'Cam Le District': 'Cam Le District',
    'Thanh Khe District': 'Thanh Khe District',
    'Ngũ Hành Sơn District': 'Ngu Hanh Son District',
    'Hoa Vang District': 'Hoa Vang District',
    'Điện Bàn District': 'Dien Ban District',
    'Lien Chiêu District': 'Lien Chieu District',
    'Liên Chieu District': 'Lien Chieu District',
    'Liên Chiểu District': 'Lien Chieu District',
    'Thang Binh District': 'Thang Binh District',
    'Sơn Trà District': 'Son Tra District',
    'unknown': 'Unknown',
    'Sơn Tra District': 'Son Tra District',
    'Không xác định': 'Unknown',
    'Lien Chiieu District': 'Lien Chieu District',
    'Thanh Khê': 'Thanh Khe District',
    'Ngy Hanh Son District': 'Ngu Hanh Son District',
    'Ngi Hanh Son District': 'Ngu Hanh Son District',
    'Hà Đông District': 'Ha Dong District',
    'Lien Chiểu District': 'Lien Chieu District',
    'Cẩm Lệ': 'Cam Le District',
    'Ng-Ngu Hanh Son District': 'Ngu Hanh Son District',
    'Not specified': 'Unknown',
    'Hoa Chau District': 'Hoa Chau District',
    'không xác định': 'Unknown',
    'Tam Ky District': 'Tam Ky District',
    'Binh Tan District': 'Binh Tan District',
    'Cẩm Le District': 'Cam Le District',
    'Cẩm Lệ District': 'Cam Le District',
    'Dien Ban District': 'Dien Ban District',
    'not specified': 'Unknown'
}

df['district'] = df['district'].map(district_mapping).fillna(df['district'])

# group by district
post_by_district = df.groupby('district').size().reset_index(name='count')

fig_district = px.bar(post_by_district, x='district', y='count', title='Post by District')
st.plotly_chart(fig_district)

# group by propertyType 
post_by_type = df.groupby('propertyType').size().reset_index(name='count')

fig_type = px.pie(post_by_type, names='propertyType', values='count', title='Post by Property Type')
st.plotly_chart(fig_type)

# select propertyType
property_types = df['propertyType'].unique()
selected_property_types = st.multiselect('Select property types', property_types, default=property_types)

# filter by selected property types
df = df[df['propertyType'].isin(selected_property_types)]

# priceBil distrbution
fig_price = px.histogram(df, x='priceBil', nbins=50, title='Price Distribution')
st.plotly_chart(fig_price)

# area distribution
fig_area = px.histogram(df, x='area', nbins=50, title='Area Distribution')
st.plotly_chart(fig_area)

# calculate pricePerM2
df['pricePerM2'] = df['priceBil'] * 1e9 / df['area']

# pricePerM2 distribution
fig_price_per_m2 = px.histogram(df, x='pricePerM2', nbins=50, title='Price per m² Distribution')
st.plotly_chart(fig_price_per_m2)

