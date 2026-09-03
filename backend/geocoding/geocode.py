from functools import partial
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="specify_your_app_name_here")

geocode = partial(geolocator.geocode, language="es")
print(geocode("london"))
Londres, Greater London, Inglaterra, SW1A 2DX, Gran Bretaña
print(geocode("paris"))
París, Isla de Francia, Francia metropolitana, Francia
print(geocode("paris", language="en"))
Paris, Ile-de-France, Metropolitan France, France

reverse = partial(geolocator.reverse, language="es")
print(reverse("52.509669, 13.376294"))
Steinecke, Potsdamer Platz, Tiergarten, Mitte, 10785, Alemania