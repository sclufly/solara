from functools import partial
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="SolarA*")

def getLocation(address: str):
    location = geolocator.geocode(address)
    return location

