import pandas as pd
import plotly.express as px
import streamlit as st
from utils.helpers import create_choropleth_map, create_parallel_coordinates_plot, create_plots, create_price_distribution_plot, create_spider_chart, load_data, load_geojson, filter_geojson, map_neighborhoods, filter_geojson_by_neighborhood, generate_wordcloud, render_table, get_airbnb_images
from streamlit_plotly_events import plotly_events
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
df, cal_df = load_data(selected_state, selected_city)

neighborhoods_geojson = load_geojson(selected_state, selected_city) if not df.empty else None

# filter neighborhood where feature.properties.Tinh_Thanh = Thành phố Đà Nẵng
neighborhoods_geojson = filter_geojson(neighborhoods_geojson, 'Tinh_Thanh', 'Thành phố Đà Nẵng')

df['neighbourhood_cleansed'] = map_neighborhoods(df, neighborhoods_geojson)

if not df.empty and neighborhoods_geojson:
    # Sidebar for neighborhood selection
    neighborhoods = ['All'] + sorted(df['neighbourhood_cleansed'].unique())

    default_neighborhood = 'All' if 'All' in neighborhoods else neighborhoods[0]
    selected_neighborhood = st.sidebar.selectbox('Select a neighborhood', neighborhoods, key='neighborhood_dropdown')

    if selected_neighborhood != 'All':
        neighborhoods_geojson = filter_geojson_by_neighborhood(neighborhoods_geojson, selected_neighborhood)
        df = df[df['neighbourhood_cleansed'] == selected_neighborhood]
        

    # price filter 0 - 2000, step 10
    price_range = st.sidebar.slider('Price range', 0, 2000, (0, 50), 10, key='price_range')
    
    # apply price filter
    df = df[(df['price'] >= price_range[0]) & (df['price'] <= price_range[1])]

    # Reset button
    # if st.button('Reset Selection'):
    #     st.session_state['selected_neighborhood'] = 'All'

    st.title(f"Airbnb : {selected_city}, {selected_state}")
    
    # group by room_type and count
    room_type_counts = df['room_type'].value_counts()
    # list of room_type with count
    room_type_with_count = map(lambda x: {'value':x[0],'label': f'{x[0]} ({x[1]})'}, room_type_counts.items())
    room_type_options = [{'value':'All','label':'All'}] + list(room_type_with_count)
    selected_room_type = st.selectbox('Select a room type', room_type_options, key='room_type_dropdown', format_func=lambda x: x['label'])
    
    if selected_room_type['value'] != 'All':
        df = df[df['room_type'] == selected_room_type['value']]
    
    # group by host 
    host_counts = df['host/id'].value_counts()
    host_with_count = map(lambda x: {'value':x[0],'label': f'{x[0]} ({x[1]})'}, host_counts.items())
    host_options = [{'value':'All','label':'All'}] + list(host_with_count)
    selected_host = st.selectbox('Select a host', host_options, key='host_dropdown', format_func=lambda x: x['label'])
    
    if selected_host['value'] != 'All':
        df = df[df['host/id'] == selected_host['value']]
    
    is_show_map = st.checkbox('Show Map', value=False)
    if is_show_map:
        choropleth_map_fig = create_choropleth_map(neighborhoods_geojson, df,selected_city, city_centers)
        selected_points = plotly_events(choropleth_map_fig, click_event=True, select_event=False, override_height=600, key="neighborhood")

    is_all_neighborhood = selected_neighborhood == 'All'
    if is_all_neighborhood:
        # analyze all neighborhood
        df_filtered = df
        # count over neighborhood
        neighborhood_counts = df['neighbourhood_cleansed'].value_counts().sort_index()
        
        # plot bar chart
        st.subheader('Neighborhood Counts')
        fig = px.bar(neighborhood_counts, x=neighborhood_counts.index, y=neighborhood_counts.values, title='Neighborhood Counts')
        st.plotly_chart(fig, use_container_width=True)
        
        # price distribution over neighborhood
        st.subheader('Price Distribution')
        fig2 = px.box(df.sort_values('neighbourhood_cleansed'), x='neighbourhood_cleansed', y='price', title='Price Distribution', color_discrete_sequence=["#FF5A5F"])
        st.plotly_chart(fig2, use_container_width=True)
        
        # avg review score over neighborhood
        st.subheader('Average Review Scores')
        fig3 = px.bar(df.groupby('neighbourhood_cleansed')['review_scores_location'].mean().sort_index().reset_index(), 
                  x='neighbourhood_cleansed', y='review_scores_location', title='Average Review Scores', color_discrete_sequence=["#FF5A5F"])
        st.plotly_chart(fig3, use_container_width=True)

    # Filter data for selected neighborhood
    if selected_neighborhood != 'All':
        df_filtered = df[df['neighbourhood_cleansed'] == selected_neighborhood]
    else:
        df_filtered = df
        
    display_charts = st.checkbox('Display Charts', value=False)
    
    if display_charts:
        st.subheader(f'Neighborhood: {selected_neighborhood}')
        
        # total locations
        st.write(f'Total Locations: {len(df_filtered)}')
        
        if not df_filtered['neighborhood_overview'].empty:
            word_cloud = generate_wordcloud(df_filtered['neighborhood_overview'])
            st.pyplot(word_cloud)
        
        fig2, fig3, fig4, fig_bar_room_type = create_plots(df_filtered,)
        col1, col2 = st.columns(2)
        
        with col1:
            create_spider_chart(df, df_filtered)
            fig5 = create_parallel_coordinates_plot(df_filtered)
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            st.plotly_chart(fig3, use_container_width=True)
            st.plotly_chart(fig_bar_room_type, use_container_width=True)
            fig6 = create_price_distribution_plot(df_filtered)
            st.plotly_chart(fig6, use_container_width=True)
        
    is_show_detail = st.checkbox('Show detail', value=False)
        
    if is_show_detail:
            ## select room to show detail
            selected_room = st.selectbox('Select a room', df_filtered['name'].values, key='room_dropdown')
            
            if selected_room:
                selected_room_df = df_filtered[df_filtered['name']==selected_room]
                display_df = render_table(selected_room_df)
                st.dataframe(display_df.T, use_container_width=True)
                                
                # thumbnail
                # st.image(selected_room_df['thumbnail'].values[0], use_container_width=True)
                
                # show all images gallery
                images = get_airbnb_images(selected_room_df)
                cols = st.columns(3)
                for i, image in enumerate(images):
                    with cols[i % 3]:
                        st.image(image, use_container_width=True)