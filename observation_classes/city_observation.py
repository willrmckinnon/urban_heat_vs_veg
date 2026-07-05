# Custom Functions
from observation_classes.general_observation import ObservedArea

# Basic Libraries
import pandas as pd


def set_date_window(lat, year, satellite):
    if satellite == 'sentinel':
        if lat >= 0: # Northern Hemisphere
            return str(f'{year}-04-01/{year}-09-29')
        else: # Southern Hemisphere
            return str(f'{year - 1}-11-01/{year}-03-28')
    else:
        if lat >= 0: # Northern Hemisphere
            return str(f'{year}-06-01/{year}-08-29')
        else: # Southern Hemisphere
            return str(f'{year - 1}-12-01/{year}-02-28')        

 
CITY_DATA_PATH = 'data/cities500.json'
CITY_DF = pd.read_json(CITY_DATA_PATH)

 
class CityObservation(ObservedArea):
    def __init__(self,
                 city_id: str,
                 collection,
                 sqkm = 250,
                 date_window: str=None,
                 year = None,
                 ):
        
        # ---------------------------
        # Find the City and extract lat/lon
        # ---------------------------   
        row = CITY_DF[CITY_DF['id'] == city_id]
        lat, lon = [row.lat.values[0], row.lon.values[0]]
        self.city_name = row.name

        # ---------------------------
        # Confirm the Date Range
        # ---------------------------    
        satellite = collection[0].split('-')[0]    
        if date_window == None:
            if year == None: year = 2025
            date_window = set_date_window(lat, year, satellite)


        super().__init__(lat, lon, sqkm, collection, date_window)








