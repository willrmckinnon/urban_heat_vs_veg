# Generic Libraries
import time
import pickle
from pathlib import Path

# Custom Libraries
from observation_classes.landsat_observation import LandsatObs
from observation_classes.sentinel_observation import SentinelObs
from operations.stats_for_multiple_cities import complete_statistical_analysis

STARTING_YEAR = 2025
EARLIEST_YEAR = 2020

class CombinedObservation():
    def __init__(self, city_id, 
                 sqkm = 250, 
                 starting_year=None, 
                 earliest_year= None, 
                 set_imgs = True, 
                 calc_stats = True,
                 incl_stats_figure = True,
                 ):
        

        if starting_year == None: starting_year=STARTING_YEAR
        if earliest_year == None: earliest_year=EARLIEST_YEAR

        self.sqkm = sqkm
        self.city_id = city_id


        # ---------------------------
        # Try Collecting the Observations
        # ---------------------------   
        landsat_obs = None
        sentinel_obs = None
        max_check = starting_year-earliest_year + 5
        cloud_thresh_params = [0.05]#, 0.1, 0.15, 0.2, 0.25]
        observations_found = False

 
        for cloud_thresh in cloud_thresh_params:
            year = starting_year 
            check = 0
            while year > earliest_year:
                check +=1
                if check > max_check: raise Exception(f'Too many attempts: Could not find a pair of observations in {check} attempts')

                try:
                    time.sleep(1)
                    landsat_obs = LandsatObs(city_id, sqkm, year=year, cloud_threshold=cloud_thresh)
                    time.sleep(1)
                    sentinel_obs = SentinelObs(city_id, sqkm, year=year, cloud_threshold=cloud_thresh)
                    break
                except Exception as e: 
                    print(e)
                    year -=1

            if landsat_obs and sentinel_obs:
                self.landsat_obs = landsat_obs
                self.sentinel_obs = sentinel_obs
                self.year = year
                observations_found = True
                break
            else: print(f'No observation found with cloud threshold of {cloud_thresh}')
        
        if not observations_found: raise Exception('Could not find a pair of observations that met requirements')


        # ---------------------------
        # Set the Attributes if Successful
        # ---------------------------   

        # Set the basic attributes
        self.ndvi = sentinel_obs.ndvi
        self.lst = landsat_obs.lst_celsius
        self.city_name = landsat_obs.city_name

        # Set the images if true
        if set_imgs: self.set_images()

        print(f'{self.city_name.iloc[0]} collection complete')



    def set_images(self):
        self.image = self.sentinel_obs.get_image()
        self.ndvi_heatmap = self.sentinel_obs.ndvi_heatmap_only()
        self.lst_heatmap = self.landsat_obs.lst_heatmap_only()

    def refresh_landsat(self, specify_year= False):
        print('Recollecting Observation')
        if specify_year: self.landsat_obs = LandsatObs(self.city_id, self.sqkm, year=self.year)
        else: self.landsat_obs = LandsatObs(self.city_id, self.sqkm)
        print('Resetting LST and Heatmap')
        self.lst = self.landsat_obs.lst_celsius
        self.lst_heatmap = self.landsat_obs.lst_heatmap_only()

    def refresh_sentinel(self, specify_year= False):
        print('Recollecting Observation')
        if specify_year: self.sentinel_obs = SentinelObs(self.city_id, self.sqkm, year=self.year)
        else: self.sentinel_obs_obs = SentinelObs(self.city_id, self.sqkm)
        print('Resetting NDVI and Heatmap')
        self.ndvi = self.sentinel_obs.ndvi
        self.ndvi_heatmap = self.sentinel_obs.ndvi_heatmap_only()

    def save(self, dest_path):
        with open(dest_path, "wb") as file:
            pickle.dump(self, file)


    def reset_lst(self):
        self.landsat_obs.set_lst()
        self.lst = self.landsat_obs.lst_celsius

    def reset_sentinel_indexes(self):
        self.sentinel_obs.set_all_indexes()
        self.ndvi = self.sentinel_obs.ndvi
        self.fvc = self.sentinel_obs.fvc
        self.savi = self.sentinel_obs.savi
        self.ndmi = self.sentinel_obs.ndmi



    def get_metadata(self):

        self.sentinel_obs.set_all_indexes()
        self.ndvi = self.sentinel_obs.ndvi
        self.fvc = self.sentinel_obs.fvc
        self.savi = self.sentinel_obs.savi
        self.ndmi = self.sentinel_obs.ndmi
        
        stats = complete_statistical_analysis(self)[0]
        if stats['city_name']: del stats['city_name']

        metadata = {
            'city_name': self.city_name.iloc[0],
            'city_id': self.city_id,
            'lst_coverage': self.landsat_obs.coverage,
            'ndvi_coverage': self.sentinel_obs.coverage,
            'ndvi_cloud_fraction': self.sentinel_obs.cloud_fraction,
            'sentinel_item_count': len(self.sentinel_obs.items),
            'landsat_item_count': len(self.landsat_obs.items),  
        }

        for k, v in stats.items(): metadata[k] = v

        return metadata

