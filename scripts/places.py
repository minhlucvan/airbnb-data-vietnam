
import uuid
import overpy

def fetch_tourirms_attraction_places(city, city_centers):
    api = overpy.Overpass()

    # Use city_centers to get the latitude and longitude
    center = city_centers.get(city, {'lat': 42.3601, 'lon': -71.0589})
    lat, lon = center['lat'], center['lon']

    query = f"""
    [out:json];
    (
    node["tourism"="attraction"](around:10000, {lat}, {lon});
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
            'type': 'attraction'
        })
    
    return places

# main

city_centers = {
    'Danang': {'lat': 16.0544, 'lon': 108.2022}
}

print('Fetching places...')
places = fetch_tourirms_attraction_places('Danang', city_centers)

print(f'Found {len(places)} places.')

# save to geojson

import json

geojson = {
    'type': 'FeatureCollection',
    'features': []
}

for place in places:
    feature = {
        'type': 'Feature',
        'id': uuid.uuid4().hex,
        'geometry': {
            'type': 'Point',
            'coordinates': [place['lon'], place['lat']]
        },
        'properties': {
            'name': place['name'],
            'type': place['type']
        }
    }
    geojson['features'].append(feature)
    
print('Saving to data/vietnam/danang/tourism.geojson...')    
with open('data/vietnam/danang/tourism.geojson', 'w') as f:
    json.dump(geojson, f)
    
print('Done.')