import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
from utils.helpers import annotate_bds_listings, create_choropleth_map, create_choropleth_map_avg, create_choropleth_map_simple, load_geojson, filter_geojson, map_neighborhoods
st.set_page_config(page_title="Airbnb Data Viz", page_icon='dashboard/  airbnb_l.svg', layout='wide', initial_sidebar_state='auto')

# st.set_page_config(layout="wide")

states = ['Vietnam', 'Massachusetts', 'New York','Illinois','Texas']

cities = {
    'Massachusetts': ['Boston', 'Cambridge'],
    'New York': ['New York City', 'Rochester'],
    'Illinois': ['Chicago'],
    'Texas': ['Austin', 'Dallas'],
    'Vietnam': ['Danang']
}
city_centers = {
    'Boston': {'lat': 42.3601, 'lon': -71.0589},
    'Cambridge': {'lat': 42.3736, 'lon': -71.1097},
    'New York City': {'lat': 40.7128, 'lon': -74.0060},
    'Rochester': {'lat': 43.161030, 'lon': -77.610924},
    'Chicago': {'lat': 41.8781, 'lon': -87.6298},
    'Austin': {'lat': 30.2672, 'lon': -97.7431},
    'Dallas': {'lat': 32.7767, 'lon': -96.7970},
    'Danang': {'lat': 16.0544, 'lon': 108.2022}
}

# Sidebar for state and city selection
selected_state = st.sidebar.selectbox('Select a state', states, key='state')
selected_city = st.sidebar.selectbox('Select a city', cities[selected_state], key='city')

# Load data and geojson based on the selections
df = pd.read_csv(f'data/{selected_state}/{selected_city}/bds.csv')

## remove all imageUrls/* columns
df = df.drop(columns=[col for col in df.columns if 'imageUrls' in col])

for row in df.itertuples():
    if not pd.isnull(df.at[row.Index, 'price']):
        if 'triệu/m²' in df.at[row.Index, 'price']:
            priceExt = df.at[row.Index, 'priceExt']
            df.at[row.Index, 'priceExt'] = df.at[row.Index, 'price']
            df.at[row.Index, 'price'] = priceExt
        elif '/m²' in df.at[row.Index, 'price']:
            priceExt = df.at[row.Index, 'priceExt']
            df.at[row.Index, 'priceExt'] = df.at[row.Index, 'price']
            df.at[row.Index, 'price'] = priceExt
        elif 'triệu' in df.at[row.Index, 'price']:
            # convert ~48,63 triệu -> 4.863 priceBil
            price = df.at[row.Index, 'price']
            priceMil = price.replace(' triệu', '').replace(',', '.')
            priceMil = float(priceMil)
            priceBil = priceMil / 1000
            priceVnd = priceMil * 1000000
            df.at[row.Index, 'priceMil'] = priceMil
            df.at[row.Index, 'priceBil'] = priceBil
            df.at[row.Index, 'priceVnd'] = priceVnd
        elif 'tỷ' in df.at[row.Index, 'price']:
            # convert ~48,63 tỷ -> 48.63 priceBil
            price = df.at[row.Index, 'price']
            priceBil = price.replace(' tỷ', '').replace(',', '.').replace('~', '')
            priceBil = float(priceBil)
            priceVnd = priceBil * 1000000000
            priceMil = priceVnd / 1000000
            df.at[row.Index, 'priceBil'] = priceBil
            df.at[row.Index, 'priceVnd'] = priceVnd
            df.at[row.Index, 'priceMil'] = priceMil
        elif 'Thỏa thuận' in df.at[row.Index, 'price']:
            df.at[row.Index, 'price'] = None
            
    if pd.notnull(df.at[row.Index, 'priceExt']):
        priceExt = df.at[row.Index, 'priceExt'].replace('~', '')
        if 'triệu/m²' in df.at[row.Index, 'priceExt']:
            # ~48,63 triệu/m² -> 48.63
            priceExt = priceExt.replace(' triệu/m²', '').replace(',', '.')
            df.at[row.Index, 'pricePerM2'] = float(priceExt)


for row in df.itertuples():
    # has price but no priceBil
    if pd.notnull(df.at[row.Index, 'price']) and pd.isnull(df.at[row.Index, 'priceBil']):
        if 'tỷ' in df.at[row.Index, 'price']:
            price = df.at[row.Index, 'price']
            priceBil = price.replace(' tỷ', '').replace(',', '.').replace('~', '')
            priceBil = float(priceBil)
            priceVnd = priceBil * 1000000000
            priceMil = priceVnd / 1000000
            df.at[row.Index, 'priceBil'] = priceBil
            df.at[row.Index, 'priceVnd'] = priceVnd
            df.at[row.Index, 'priceMil'] = priceMil
            
# propertyType = 'Land' if bedroom is NaN else 'House'
df['propertyType'] = df.apply(lambda x: 'Land' if pd.isnull(x['bedroom']) else 'House', axis=1)
        
# drop row if both price and priceExt are NaN
df = df.dropna(subset=['price', 'priceExt'], how='all')

# group by propertyType
propertyType = df.groupby('propertyType').size().reset_index(name='count')
fig = px.pie(propertyType, values='count', names='propertyType', title='Property Type Distribution')
st.plotly_chart(fig)

# select propertyType
selected_propertyType = st.selectbox('Select a property type', ['House', 'Land'])

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


neighborhoods_geojson = load_geojson(selected_state, selected_city) if not df.empty else None

# filter neighborhood where feature.properties.Tinh_Thanh = Thành phố Đà Nẵng
neighborhoods_geojson = filter_geojson(neighborhoods_geojson, 'Tinh_Thanh', 'Thành phố Đà Nẵng')

# df with lat, lon
df['coordinates/latitude'] = df['lat'].astype(float)
df['coordinates/longitude'] = df['long'].astype(float)

# drop rows with missing coordinates
df = df.dropna(subset=['coordinates/latitude', 'coordinates/longitude'])

df['neighbourhood_cleansed'] = map_neighborhoods(df, neighborhoods_geojson)

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