from shapely.geometry import Point, Polygon
from geopy.distance import geodesic

def distance_to_polygon(lon, lat, geometry):
    point = Point(lon, lat)
    # coordinates: list of list of float
    coordinates = [tuple(coord) for coord in geometry['coordinates'][0]]
    polygon = Polygon(coordinates)
    distance_in_degrees = polygon.distance(point)
    
    # Convert distance from degrees to kilometers
    distance_in_km = geodesic((lat, lon), (lat + distance_in_degrees, lon)).km
    return distance_in_km


def calculate_distance_to_coast(lon, lat, coast_geojson):
    # type feature collection
    if coast_geojson['type'] == 'FeatureCollection':
        # type feature
        min_distance = None
        if coast_geojson['features'][0]['type'] == 'Feature':
            # type geometry
            if coast_geojson['features'][0]['geometry']['type'] == 'Polygon':
                for polygon in coast_geojson['features']:
                    geometry = polygon['geometry']
                    distance = distance_to_polygon(lon, lat, geometry)
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
        return min_distance
    
    return None

# add distance to coast to the dataframe
def calc_distance_to_coast(df, coast_geojson):
    distance_to_coast = df.apply(lambda x: calculate_distance_to_coast(x['lon'], x['lat'], coast_geojson), axis=1)
    return distance_to_coast

# Haversine formula to calculate distance between two points
def haversine(lon1, lat1, lon2, lat2):
    import math
    R = 6371
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def calc_distance_to_point(df, lon, lat):
    distance_to_point = df.apply(lambda x: haversine(lon, lat, x['lon'], x['lat']), axis=1)
    return distance_to_point

def find_place_by_name(tourism_geojson, name):
    for feature in tourism_geojson['features']:
        if feature['properties']['name'] == name:
            return feature
    return None