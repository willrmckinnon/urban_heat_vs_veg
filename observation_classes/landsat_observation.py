# Custom Functions
from observation_classes.city_observation import CityObservation

# Basic Libraries
import gc
import odc.stac
import numpy as np
import xarray as xr
from PIL import Image
import geopandas as gpd
import planetary_computer
from shapely.geometry import shape
from matplotlib import pyplot as plt

 
class LandsatObs(CityObservation):
    def __init__(self, 
                 city_id: str, 
                 sqkm = 250, 
                 date_window: str=None, 
                 year=None,
                 cloud_threshold = 0.1,
                 ):
        collection = ["landsat-c2-l2"]
        self.cloud_threshold = cloud_threshold
        super().__init__(city_id, collection, sqkm, date_window=date_window, year=year)

        self.set_lst()


    #Selects only the most recent item from each MGRS tile using Landsat Configuration
    def filter_items(self, items):
        """
        1. Filter by Cloud Coverage
        2. Select only the latest of each row/path combination
        """

        cloudless_items = []
        for item in items:
            ds = self.stack(['qa_pixel'],[item])

            qa = ds["qa_pixel"]
            cloud_mask = (qa.astype("uint16") & (1 << 3)) > 0
            cloud_fraction = cloud_mask.mean(dim=["x","y"])
            cloud_fraction = cloud_fraction.compute().item()

            if cloud_fraction < self.cloud_threshold: cloudless_items.append(item)    

        #Only take the most recent item of each row/path tile        
        latest_items = {}
        for item in cloudless_items:
            # Get the Grid
            path = int(item.properties.get("landsat:wrs_path"))
            row = int(item.properties.get("landsat:wrs_row"))
            tile_id = f"{path:03d}_{row:03d}"

            if tile_id and tile_id not in latest_items:
                latest_items[tile_id] = item


        return list(latest_items.values())    






    #Method to stack specified bands of the observation's items
    #RETURNS: Numpy Array and X Array
    def stack(self, bands, items = None, aoi = None):
        """
        1. Stack the bands
        2. For each band, combine the data to produce a single view with the most possible coverage
        """

        if items == None: items = self.items
        if aoi == None: aoi = self.aoi
        #Sign the items
        signed_items = []
        for item in items: signed_items.append(planetary_computer.sign(item))
        
        #collect the xarray
        xx = odc.stac.load(
            signed_items,
            bands = bands,
            geopolygon=aoi,
            resampling = 'bilinear',
            chunks = {'x': 512, 'y': 512}
        )

        xx = xx[bands].astype("uint16")


        composite_ds = xr.Dataset()

        for band_name in bands:
            band = xx[band_name]
            #Splice together the time dimensions to get the most complete view
            # Convert 0 pixels to nan
            band = band.where(band != 0)

            # Calculate the coverage of each time dim and sort
            coverage = []
            for i in range(len(band.time)):
                scene = band.isel(time=i)
                valid_pixels = scene.notnull().sum().compute()
                coverage.append(valid_pixels)

            order = np.argsort(coverage)[::-1]

            # Splice together scenes in order of most coverage
            composite = band.isel(time=order[0])
            for idx in order[1:]:
                composite = composite.combine_first(band.isel(time=idx))

            composite_ds[band_name] = composite
        return composite_ds




    # Function to generate land surface temperature
    def set_lst(self):
        lst_band = ['lwir11']
        lst_raw = self.stack(lst_band)

        self.lst_kelvin = lst_raw * 0.00341802 + 149.0
        self.lst_celsius = self.lst_kelvin - 273.15

        #self.lst_kelvin = lst_kelvin.load()
        #self.lst_celsius = lst_celsius.load()

        # Calculate the coverage
        lst_arr = self.lst_celsius.to_array().to_numpy().squeeze()
        total = lst_arr.size
        nan_pix = np.isnan(lst_arr).sum()
        self.coverage = 1 - (nan_pix/total)


    # Function to return a heatmap of lst
    def lst_heatmap_only(self, cmap='hot'):
        lst_arr = self.lst_celsius.to_array().to_numpy().squeeze()
        lst_arr[lst_arr<0]=0

        # Handle any nan values
        temp_arr = np.nan_to_num(lst_arr, nan=0)
        lst_arr = lst_arr/(np.nanmax(temp_arr)+1e-6)

        # Apply colormap from plt
        rgba = plt.get_cmap(cmap)(lst_arr)

        # Convert to uint8 RGB
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb)
