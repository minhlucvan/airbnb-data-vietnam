import pandas as pd
import plotly.express as px
import streamlit as st
import json

def get_amenities(row):
    amenities = ''
    total_amenities = 100
    # amenities/0/values/2/available
    # amenities/0/values/2/title
    # amenities/1/values/2/icon
    for i in range(1, total_amenities):
        ammenity_title = f'amenities/{i}/title'
        amenities_available = f'amenities/{i}/available'
        
    return amenities

# Function to load data based on selected state and city
def load_data(state, city):
    filename = f'data/{state.lower()}/{city.lower()}/listings.csv'
    cal_filename = f'data/{state.lower()}/{city.lower()}/calendar.csv'
    try:
        df = pd.read_csv(filename)
        # remove all colums images/*
        # df = df.loc[:, ~df.columns.str.contains('^images/')]
        
        # remove all colums houseRules/*
        # df = df.loc[:, ~df.columns.str.contains('^houseRules/')]
        
        # remove all colums breadcrumbs
        # df = df.loc[:, ~df.columns.str.contains('^breadcrumbs/')]
        
        df['price'] = df['price/price']
        df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
        
        # st.write(df.columns)
        # androidLink
        # brandHighlights/hasGoldenLaurel
        # brandHighlights/subtitle
        # brandHighlights/title
        # cancellationPolicies/0/policyId
        # cancellationPolicies/0/policyName
        # cancellationPolicies/0/title
        # cancellationPolicies/1/policyId
        # cancellationPolicies/1/policyName
        # cancellationPolicies/1/title
        # checkIn
        # checkOut
        # coHosts/0/name
        # coHosts/0/profilePictureUrl
        # coHosts/0/userId
        # coHosts/1/name
        # coHosts/1/profilePictureUrl
        # coHosts/1/userId
        # coHosts/2/name
        # coHosts/2/profilePictureUrl
        # coHosts/2/userId
        # coHosts/3/name
        # coHosts/3/profilePictureUrl
        # coHosts/3/userId
        # coordinates/latitude
        # coordinates/longitude
        # description
        # descriptionOriginalLanguage
        # highlights/0/icon
        # highlights/0/subtitle
        # highlights/0/title
        # highlights/0/type
        # highlights/1/icon
        # highlights/1/subtitle
        # highlights/1/title
        # highlights/1/type
        # highlights/2/icon
        # highlights/2/subtitle
        # highlights/2/title
        # highlights/2/type
        # homeTier
        # host
        # host/about
        # host/highlights/0
        # host/highlights/1
        # host/highlights/2
        # host/highlights/3
        # host/highlights/4
        # host/hostDetails/0
        # host/hostDetails/1
        # host/id
        # host/isSuperHost
        # host/isVerified
        # host/name
        # host/profileImage
        # host/ratingAverage
        # host/ratingCount
        # host/timeAsHost/months
        # host/timeAsHost/years
        # houseRules
        # htmlDescription/htmlText
        # htmlDescription/recommendedNumberOfLines
        # id
        # iosLink
        # language
        # location
        # locationDescriptions/0/content
        # locationDescriptions/0/mapMarkerRadiusInMeters
        # locationDescriptions/0/title
        # locationDescriptions/1/content
        # locationDescriptions/1/mapMarkerRadiusInMeters
        # locationDescriptions/1/title
        # locationDescriptions/2/content
        # locationDescriptions/2/mapMarkerRadiusInMeters
        # locationDescriptions/2/title
        # locationSubtitle
        # metaDescription
        # personCapacity
        # price/breakDown/basePrice/description
        # price/breakDown/basePrice/price
        # price/breakDown/cleaningFee
        # price/breakDown/cleaningFee/description
        # price/breakDown/cleaningFee/price
        # price/breakDown/earlyBirdDiscount
        # price/breakDown/earlyBirdDiscount/description
        # price/breakDown/earlyBirdDiscount/price
        # price/breakDown/serviceFee
        # price/breakDown/serviceFee/description
        # price/breakDown/serviceFee/price
        # price/breakDown/specialOffer
        # price/breakDown/specialOffer/description
        # price/breakDown/specialOffer/price
        # price/breakDown/taxes
        # price/breakDown/taxes/description
        # price/breakDown/taxes/price
        # price/breakDown/total
        # price/breakDown/total/description
        # price/breakDown/total/price
        # price/breakDown/totalBeforeTaxes
        # price/discountedPrice
        # price/label
        # price/originalPrice
        # price/price
        # price/qualifier
        # propertyType
        # rating
        # rating/accuracy
        # rating/checking
        # rating/cleanliness
        # rating/communication
        # rating/guestSatisfaction
        # rating/location
        # rating/reviewsCount
        # rating/value
        # roomType
        # seoTitle
        # sharingConfigTitle
        # subDescription/items/0
        # subDescription/items/1
        # subDescription/items/2
        # subDescription/items/3
        # subDescription/title
        # thumbnail
        # timestamp
        # title
        # url
        # price
        # price/price     
        
        # neighbourhood = locationDescriptions/0/title
        df['neighbourhood'] = df['locationDescriptions/0/title']
        
        # review_scores_location = rating/location
        df['review_scores_location'] = df['rating/location']
        
        # neighborhood_overview = description
        df['neighborhood_overview'] = df['description']
        
        # review_scores_rating = rating
        df['review_scores_rating'] = df['rating']
        
        # availability_30 = availability/availability_30
        df['availability_30'] = 0
        df['availability_60'] = 0
        df['availability_90'] = 0
        df['availability_365'] = 0
        df['room_type'] = df['roomType']
        
        # KeyError: "['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_value'] not in index"
        df['review_scores_accuracy'] = df['rating/accuracy']
        df['review_scores_cleanliness'] = df['rating/cleanliness']
        df['review_scores_checkin'] = df['rating/checking']
        df['review_scores_communication'] = df['rating/communication']
        df['review_scores_value'] = df['rating/value']
        
        # KeyError: "['host_is_superhost', 'accommodates'] not in index"
        df['host_is_superhost'] = df['host/isSuperHost']
        df['accommodates'] = df['personCapacity']
        
        # bathrooms_text
        df['bathrooms_text'] = ''
        
        # KeyError: "['listing_url', 'name', 'picture_url', 'host_name', 'host_picture_url', 'property_type', 'beds'] not in index"
        df['listing_url'] = df['url']
        df['name'] = df['title']
        df['picture_url'] = df['thumbnail']
        df['host_name'] = df['host/name']
        df['host_picture_url'] = df['host/profileImage']
        df['property_type'] = df['propertyType']
        df['beds'] = 0
        df['bath'] = 0
        
        df['lat'] = df['coordinates/latitude']
        df['lon'] = df['coordinates/longitude']
        
        df['name'] = df['title']
        
        # price per person
        df['price_per_person'] = df['price'] / df['accommodates']
                
        cal_df = pd.read_csv(cal_filename)
        return df,cal_df
    except FileNotFoundError:
        st.error(f"Data file not found: {filename},{cal_filename}")
        return pd.DataFrame()

# Function to load GeoJSON based on selected state and city
def load_geojson(state, city):
    geojson_path = f'data/{state.lower()}/{city.lower()}/neighbourhoods.geojson'
    try:
        with open(geojson_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON file not found: {geojson_path}")
        return None
    
def load_geojson_from_path(geojson_path):
    try:
        with open(geojson_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"GeoJSON file not found: {geojson_path}")
        return None

def filter_geojson(geojson, key, value):
    if geojson:
        filtered_features = [feature for feature in geojson['features'] if feature['properties'][key] == value]
        geojson['features'] = filtered_features
        return geojson
    return None

def filter_geojson_by_neighborhood(geojson, neighborhood):
    return filter_geojson(geojson, 'name', neighborhood)

# Function to check if a point is inside a polygon
def is_point_in_polygon(polygon, latitude, longitude):
    is_inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if y1 == y2:
            continue
        if min(y1, y2) < latitude <= max(y1, y2):
            x_intersection = (x2 - x1) / (y2 - y1) * (latitude - y1) + x1
            if x_intersection < longitude:
                is_inside = not is_inside
    return is_inside

# search feature by cordination
def search_feature_by_cordination(geojson, latitude, longitude):
    for feature in geojson['features']:
        if feature['geometry']['type'] == 'Polygon':
            if is_point_in_polygon(feature['geometry']['coordinates'][0], latitude, longitude):
                return feature
        elif feature['geometry']['type'] == 'MultiPolygon':
            for polygon in feature['geometry']['coordinates']:
                if is_point_in_polygon(polygon[0], latitude, longitude):
                    return feature
    return None

# map each listing to a neighborhood
# by geojson
# df['cordinates/latitude'], df['cordinates/longitude'] insde geojson polygon
def map_neighborhoods(df, geojson):
    if geojson:
        neighborhoods = []
        for i, row in df.iterrows():
            feature = search_feature_by_cordination(geojson, row['coordinates/latitude'], row['coordinates/longitude'])
            if feature:
                neighborhoods.append(feature['properties']['name'])
            else:
                neighborhoods.append('Unknown')
        return neighborhoods
    return

import overpy

def fetch_museum_artwork_places(city, city_centers):
    api = overpy.Overpass()

    # Use city_centers to get the latitude and longitude
    center = city_centers.get(city, {'lat': 42.3601, 'lon': -71.0589})
    lat, lon = center['lat'], center['lon']

    # Fetch museum and artwork places
    query = f"""
    [out:json];
    (
    node["tourism"="museum"](around:20000,{lat},{lon});
    node["leisure"="park"](around:20000,{lat},{lon});
    node["historic"="monument"](around:20000,{lat},{lon});
    node["historic"="castle"](around:20000,{lat},{lon});
            );
    out center;
    """

    result = api.query(query)

    # Collecting museum and artwork places
    places = []
    for node in result.nodes:
        places.append({
            'name': node.tags.get('name', 'Unnamed'),
            'lat': float(node.lat),
            'lon': float(node.lon),
            # 'type': node.tags.get('tourism')
            'type': node.tags.get('tourism', node.tags.get('historic', node.tags.get('leisure', node.tags.get('religion', 'Unknown'))))

        })

    return places

def fetch_museum_artwork_places_cached(city, city_centers):
    filename = f'data/{city.lower()}_places.json'
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        places = fetch_museum_artwork_places(city, city_centers)
        with open(filename, 'w') as f:
            json.dump(places, f)
        return places

import plotly.graph_objs as go

def create_choropleth_map_avg(neighborhoods_geojson, data, metric, city, city_centers):#, city_centers, neighborhoods_geojson):
    avg_prices = data.groupby('neighbourhood_cleansed')[metric].mean().reset_index()
    min_val = avg_prices[metric].min() - 0.1
    
    for feature in neighborhoods_geojson['features']:
        neighborhood = feature['properties']['name']
        price = avg_prices[avg_prices['neighbourhood_cleansed'] == neighborhood][metric]
        feature['properties']['avg_price'] = price.values[0] if not price.empty else None
        
    center = city_centers.get(city, {'lat': 42.3601, 'lon': -71.0589})
    
    fig = px.choropleth_mapbox(
        avg_prices,
        geojson=neighborhoods_geojson,
        locations='neighbourhood_cleansed',
        featureidkey="properties.name",
        color=metric,
        color_continuous_scale="bugn",
        range_color=[min_val, 500],
        mapbox_style="carto-positron",
        zoom=10,
        center=center,
        opacity=0.5,
        labels={'price': 'Price'}
    )   
    
    fig.update_layout(geo_bgcolor='rgba(0,0,0,0)')
    
    return fig
    

def create_choropleth_map_simple(hexagon_geojson, df, city, city_centers):#, city_centers, hexagon_geojson):
    
    # fake data by hexagon_geojson features
    data = pd.DataFrame(hexagon_geojson['features'])
    
    center = city_centers.get(city, {'lat': 42.3601, 'lon': -71.0589})
    fig = px.choropleth_mapbox(
        data,
        geojson=hexagon_geojson,
        locations='id',
        featureidkey="id",
        # color='price',
        color_continuous_scale="bugn",
        # range_color=[0, 500],
        mapbox_style="carto-positron",
        zoom=10,
        center=center,
        opacity=0.5,
        # labels={'price': 'Price'}
    )
    fig.update_layout(
        margin={"r":0, "t":0, "l":0, "b":0},
        legend=dict(
            title='Place Types',
            orientation='v',
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.01  
        )
    )

    fig.update_layout(geo_bgcolor='rgba(0,0,0,0)')
    
    return fig

def create_choropleth_map(neighborhoods_geojson, data, city, city_centers):#, city_centers, neighborhoods_geojson):        
    avg_scores = data.groupby('neighbourhood_cleansed')['review_scores_location'].mean().reset_index()
    min_val = avg_scores['review_scores_location'].min() - 0.1        
    
    for feature in neighborhoods_geojson['features']:
        neighborhood = feature['properties']['name']
        score = avg_scores[avg_scores['neighbourhood_cleansed'] == neighborhood]['review_scores_location']
        feature['properties']['avg_review_score_location'] = score.values[0] if not score.empty else None

    center = city_centers.get(city, {'lat': 42.3601, 'lon': -71.0589}) 
    
    fig = px.choropleth_mapbox(
        avg_scores,
        geojson=neighborhoods_geojson,
        locations='neighbourhood_cleansed',
        featureidkey="properties.name",
        color='review_scores_location',
        color_continuous_scale="bugn",
        range_color=[min_val, 5],
        mapbox_style="carto-positron",
        # mapbox_style="basic",
        zoom=10,
        center=center,
        opacity=0.3,
        labels={'review_scores_location': 'Location Ratings'}
    )
    fig.update_layout(geo_bgcolor='rgba(0,0,0,0)')
    
    ## add museum and artwork places
    fig = annotate_museum_artwork_places(fig, city, city_centers)
    
    ## Add makrers for each listing
    fig = annotate_airbnb_listings(fig, data, city)

    fig.update_layout(
        margin={"r":0, "t":0, "l":0, "b":0},
        legend=dict(
            title='Place Types',
            orientation='v',
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.01  
        )
    )

    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
    return fig


def create_availability_plot(dataframe):
    availability_metrics = dataframe.groupby('neighbourhood_cleansed').agg({
        'availability_30': lambda x: x.mean() / 30 * 100,
        'availability_60': lambda x: x.mean() / 60 * 100,
        'availability_90': lambda x: x.mean() / 90 * 100,
        'availability_365': lambda x: x.mean() / 365 * 100
    }).reset_index()

    availability_long = availability_metrics.melt(id_vars=['neighbourhood_cleansed'], var_name='Availability Period', value_name='Average Availability Percentage')

    fig = px.bar(
        availability_long,
        x='neighbourhood_cleansed',
        y='Average Availability Percentage',
        color='Availability Period',
        title='Average Availability Percentage by Neighbourhood',
        labels={'neighbourhood_cleansed': 'Neighbourhood'},
        barmode='group'
    )

    return fig    

def create_room_type_bar_plot(data):
    
    room_types = ['Private room', 'Entire home/apt', 'Shared room', 'Hotel room']

    room_type_counts = data['room_type'].value_counts().reindex(room_types, fill_value=0).reset_index()
    room_type_counts.columns = ['room_type', 'count']

    # Define a representative Airbnb color
    airbnb_color = '#FF5A5F'

    fig = px.bar(room_type_counts, x='room_type', y='count', 
                title='Room Type Distribution',
                labels={'count': 'Count', 'room_type': 'Room Type'},
                color_discrete_sequence=[airbnb_color])

    fig.update_layout(showlegend=False)
    return fig

def create_histogram(dataframe):
    # Define the number of bins, or alternatively, set the range and size of each bin
    nbins = 20  # For example, 20 bins
    range_x = [0, 5]  # Assuming the review scores range from 0 to 10
    bin_size = 0.5  # Each bin will have a size of 0.5

    fig = px.histogram(dataframe, x='review_scores_rating', nbins=nbins,
                    title='Distribution of Review Scores',
                    labels={'review_scores_rating': 'Review Score Rating'},
                    color_discrete_sequence=["#FF5A5F"],
                    range_x=range_x,
                    histnorm='percent')  # Optional: normalize to show percentages

    return fig    

# Define function to create plots for the selected neighborhood
def create_plots(dataframe):
    fig2 = px.box(dataframe, x='neighbourhood_cleansed', y='price', title='Price Distribution', color_discrete_sequence=["#FF5A5F"])
    fig3 = create_histogram(dataframe)        
    fig4 = create_availability_plot(df_filtered)
    fig5 = create_room_type_bar_plot(df_filtered)
    return fig2, fig3, fig4, fig5

def render_table(df_filtered):
    columns = [
            'name',
            # 'picture_url',
            'host_name',
            # 'host_picture_url',
            'host_is_superhost',
            'neighbourhood_cleansed', 
            'property_type',
            'room_type',
            'accommodates', 
            'bath',
            'beds',
            'price',
            'listing_url',
            ]
    
    df_display = df_filtered[columns]

    return df_display

def create_spider_chart(df, df_filtered):
    categories = ['review_scores_accuracy', 'review_scores_cleanliness', 
                        'review_scores_checkin', 'review_scores_communication', 
                        'review_scores_location', 'review_scores_value']    
    display_categories = [category.replace('review_scores_', '').replace('_', ' ').title() for category in categories]

    # Calculate the mean scores for each review category for the entire dataset
    mean_scores = df[categories].mean().reset_index(name='Value')
    print(mean_scores)
    mean_scores['Type'] = selected_city #'Overall'
    mean_scores['Variable'] = display_categories


    # Calculate the mean scores for each review category for the filtered dataset (specific neighborhood)
    mean_scores_filtered = df_filtered[categories].mean().reset_index(name='Value')
    mean_scores_filtered['Type'] = selected_neighborhood
    mean_scores_filtered['Variable'] = display_categories

    # Combine the two dataframes
    # print(radar_data.columns)
    radar_data = pd.concat([mean_scores, mean_scores_filtered])
    min_val,max_val = min(radar_data['Value'])-0.5, 5

    print(radar_data['Value'],min(radar_data['Value']))
    radar_data.columns = ['Index', 'Value', 'Type','Variable']
    print(mean_scores_filtered,mean_scores)
    # Create the radar chart
    fig = px.line_polar(radar_data, r='Value', theta='Variable', color='Type', line_close=True,
                        color_discrete_sequence=px.colors.qualitative.D3, # Using a predefined color sequence
                        template="plotly_white", # Using a light theme that fits well with most Streamlit themes
                        range_r=[min_val,max_val],title='Radar plot for Review scores')

    # Update the layout to make it cleaner
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min_val,max_val]
            )),
        showlegend=True
    )
    # Add fill with translucency
    fig.update_traces(fill='toself', fillcolor='rgba(0,100,200,0.2)') # Adjust RGBA values as needed

    # Show the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    df[['accommodates','price']] = df[['accommodates','price']].astype({'accommodates':int,'price':float}) 
    with modification_container:
        # to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        to_filter_columns = ['Room Type','Price','Host is superhost?','Beds','Accommodates','Bath']#,
    
    df.rename(columns={'room_type':'Room Type','host_is_superhost':'Host is superhost?','accommodates':'Accommodates','bath':'Bath','beds':'Beds','price':'Price'},inplace=True)

    for i in range(0, len(to_filter_columns), 2):
        cols = st.columns(2)  # Create two columns
        for j, column in enumerate(to_filter_columns[i:i+2]):
            with cols[j]:
                # st.write("↳")
                # print(df[column].unique())
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = st.multiselect(
                        # f"Values for {column}",
                        f"{column}:",

                        list(sorted(df[column].unique())),
                        default=sorted(list(df[column].unique())),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = st.slider(
                        # f"Values for {column}",
                        f"{column}",
                        _min,
                        _max,
                        (_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = st.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = st.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].str.contains(user_text_input)]

    return df

def create_parallel_coordinates_plot(data):
    # Define the columns to be included in the plot
    # Ensure that 'price' is at the end to serve as the last axis
    columns = [
        'host_is_superhost',
        'review_scores_rating','accommodates', 'price'
    ]#,'number_of_reviews'

    # Filter the dataframe to include only the relevant columns
    filtered_data = data[columns].dropna()
    filtered_data['host_is_superhost'] = filtered_data['host_is_superhost'].apply(lambda x: 1 if x=='t' else 0)

    # Define the ranges for each dimension, specifically setting price's minimum to 0
    dimensions = [
        dict(range=[filtered_data['host_is_superhost'].min(), filtered_data['host_is_superhost'].max()], label='Is Superhost', values=filtered_data['host_is_superhost']),
        dict(range=[filtered_data['review_scores_rating'].min(), filtered_data['review_scores_rating'].max()], label='Review Score', values=filtered_data['review_scores_rating']),
        dict(range=[0, filtered_data['price'].max()], label='Price', values=filtered_data['price'])
    ]
    filtered_data= pd.concat([filtered_data,pd.DataFrame([{'host_is_superhost':0,'review_scores_rating':0,'price':0}])],ignore_index=True)
    
    # Create the parallel coordinates plot
    fig = px.parallel_coordinates(filtered_data, color="price",
                                labels={
                                    "host_is_superhost":'Superhost',
                                    "review_scores_rating": "Review Score",
                                    "price": "Price"
                                },
                                title="Parallel Plots")
    fig.update_layout(
            margin=dict(l=50)  # Adjust left margin to prevent cutoff , t=50, b=50
        )
    fig.update_traces(line=dict(color='white', width=0.5), selector=dict(mode='lines'))

    
    return fig


import calplot
import matplotlib.pyplot as plt
import io
def create_neighborhood_calendar_heatmap(df_display_in, cal_df_in, st,neigborhood):

    if type(neigborhood)==str:
        formatted_neighborhood = neigborhood.lower().replace(' ', '_').replace('/', '_')
    else:
        formatted_neighborhood = str(neigborhood).lower().replace(' ', '_').replace('/', '_')


    neighborhood_dir = f'data/{selected_state.lower()}/{selected_city.lower()}/calendar/'

    neighborhood_cal_df = pd.read_csv(f'{neighborhood_dir}{formatted_neighborhood}.csv')
    neighborhood_cal_df['date'] = pd.to_datetime(neighborhood_cal_df['date'])
    neighborhood_cal_df = neighborhood_cal_df[neighborhood_cal_df['date'] > pd.to_datetime('today')]
    data_for_plot = neighborhood_cal_df.set_index('date')['available']

    fig, ax = calplot.calplot(data_for_plot, cmap='OrRd', suptitle=f'{neigborhood} Availability Calendar Heatmap', figsize=(15, 3))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    st.image(buf, use_column_width=True)

    buf.close()
    plt.close(fig)
    
def create_price_distribution_plot(data):
    fig = px.histogram(data, x='price', title='Price Distribution', color_discrete_sequence=["#FF5A5F"])
    return fig

from wordcloud import WordCloud
import re
def generate_wordcloud(data):
    # Join all the entries in the neighborhood overview column into a single text
    text = ' '.join(description for description in data if description and not pd.isnull(description))
    text = re.sub(r'<.*?>', '', text)
    words_to_remove = ['neighborhood', 'neighbourhood',f'{selected_neighborhood}','street','one','city']  # ,selected_city,selected_state Add words to remove
    
    for word in words_to_remove:
        text = text.replace(word, '')        
    wordcloud = WordCloud(width = 800, height = 300, background_color ='white').generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  
    plt.show()
    
    return fig

def annotate_museum_artwork_places(fig, city, city_centers):
    color_map = {
        'museum': 'gold',
        'monument': 'gold',
        'theatre': 'purple',
        'park': 'darkgreen',  # Changed from darkgreen to brown
        'gallery': 'orange',
        'religion': 'gold',
        'Unknown': 'gray',  # For any unclassified or missing types
        'artwork': 'magenta',  # Changed from green to magenta

    }
    
    # Fetch museum and artwork places
    tourism_places = fetch_museum_artwork_places_cached(city, city_centers)

    # Organize data by type for legend management
    for place_type, color in color_map.items():
        filtered_places = [p for p in tourism_places if p['type'] == place_type]
        if filtered_places:  # Only add traces if there are places of this type
            fig.add_trace(
                go.Scattermapbox(
                    lat=[p['lat'] for p in filtered_places],
                    lon=[p['lon'] for p in filtered_places],
                    mode='markers+text',
                    marker=go.scattermapbox.Marker(
                        size=9,
                        color=color
                    ),
                    text=[p['name'] for p in filtered_places],
                    textposition='bottom right',
                    name=place_type.capitalize(),  # Use the type as the name for the legend
                    showlegend=True  # Enable legend for this trace
                )
            )

    return fig

def annotate_airbnb_listings(fig, data, city):
    fig.add_trace(
        go.Scattermapbox(
            lat=data['coordinates/latitude'],
            lon=data['coordinates/longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=9,
                color=data['price'],
                colorscale='Bluered',
                colorbar=dict(title="Price"),
                opacity=0.5
            ),
            text=data['name'],
            name='Airbnb Listings',
            showlegend=True
        )
    )
    
    return fig

def annotate_bds_listings(fig, data, city):
    fig.add_trace(
        go.Scattermapbox(
            lat=data['coordinates/latitude'],
            lon=data['coordinates/longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=9,
                color=data['pricePerM2'],
                colorscale='Bluered',
                colorbar=dict(title="Price"),
                opacity=0.5
            ),
            text=data['title'],
            name='BDS Listings',
            showlegend=True
        )
    )
    
    return fig

def get_airbnb_images(df):
    images = []
    # extract all images/n/imageUrl
    for i in range(1, 30):
        image_url = f'images/{i}/imageUrl'
        if image_url in df.columns:
            url_values = df[image_url].values
            if not pd.isnull(url_values[0]):
                images.extend(url_values)
    
    return images
