# Custom Functions
from observation_classes.city_observation import CityObservation

# Basic Libraries
import gc
import odc.stac
import numpy as np
import pandas as pd
from PIL import Image
import geopandas as gpd
import rioxarray as rio
import planetary_computer
from shapely.geometry import shape
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon



MGRS_REFERENCE_FILE = 'data/copernicus_tiling_reference.kml'

# Function needed to handle polygons with z data
def drop_z(geom):
    if geom.geom_type == "Polygon":
        return Polygon([(x, y) for x, y, *_ in geom.exterior.coords])
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([
            Polygon([(x, y) for x, y, *_ in p.exterior.coords])
            for p in geom.geoms
        ])
    return geom
    

 
 
class SentinelObs(CityObservation):
    def __init__(self, 
                 city_id: str, 
                 sqkm = 250, 
                 date_window: str=None, 
                 year=None,
                 cloud_threshold = 0.1,
                 ):
        collection = ["sentinel-2-l2a"]
        self.cloud_threshold = cloud_threshold
        self.cloud_fraction = None
        super().__init__(city_id, collection, sqkm, date_window=date_window, year=year)
        self.mgrs_gdf = None
        
        
        # Set the NDVI
        self.set_ndvi()




    #Selects only the most recent item from each MGRS tile using Sentinel Configuration
    def filter_items(self, items):
        """
        1. Separate items into their mgrs buckets
        2. for all mgrs, define the geometry that overlaps the mgrs and aoi
        3. for all mgrs, test each items coverage of that overlapping area
        4. Keep the best item for each mgrs
        """
        
        #Sort the items with the earliest first
        items = sorted(items, key=lambda item: item.datetime, reverse=True)

        item_grids = {}
        for item in items:
            tile_id = item.properties.get("s2:mgrs_tile")
            if tile_id and tile_id not in item_grids:
                item_grids[tile_id] = [item]
            else: 
                item_grids[tile_id].append(item)

        self.setup_mgrs_ref(list(item_grids.keys()))

        for grid in item_grids.keys():
            # Establish the intersection of the aoi and the grids area
            geom = shape(self.mgrs_gdf.loc[grid].geometry)
            grid_geom = geom.intersection(self.aoi)
            grid_geom = drop_z(grid_geom)

            items_to_remove = []
            for item in item_grids[grid]:
                # Stack the item
                item_stac = self.stack(['red', 'SCL'], items=[item], aoi=grid_geom)

                # Check for data coverage
                xx = item_stac[1]
                nan_count = int(xx.red.isnull().sum().compute())
                tot_count = xx.red.size
                coverage = 1 - (nan_count/tot_count)

                # Check for cloud cover
                scl = item_stac[0][:,:,1]
                cloud_mask = np.isin(scl, [1, 3, 8, 9, 10,11]).astype('int64')
                cloud_fraction = cloud_mask.mean()

                # If the item doesnt cover the area that the grid is responsible for, or  
                # it has too many clouds, it is set to be removed
                if coverage < 0.999 or cloud_fraction > self.cloud_threshold: items_to_remove.append(item)
            
            # Remove the items
            for item in items_to_remove: item_grids[grid].remove(item)

            if len(item_grids[grid]) < 1: raise ValueError('One of the MGRS grids did not have an item that covered the required area')

        # Only keep the most recent valid items from each grid
        final_items = []
        for k in item_grids.keys(): final_items.append(item_grids[k][-1])

        return final_items


    #Method to quickly return the visual as a PIL image
    def get_image(self, mask_type = None, target_sat = 75):

        data = self.stack(['B02','B03','B04'])[0]
        norm_data = np.zeros(data.shape)


        for i in range(data.shape[2]):
            band = data[:,:,i]
            band = np.nan_to_num(band,copy=False)
            band = (band - band.min()) / (band.max() - band.min())
            band = (255 * band).astype(np.uint8)
            norm_data[:,:,i] = band
        norm_data = norm_data[:,:,[2,1,0]]


        rat = target_sat/norm_data.mean()
        #exp = ln(target_sat)/(ln(norm_data.mean())+1e-6)
        #norm_data = np.power(norm_data, exp)
        norm_data = np.clip((norm_data*rat),0,255).astype(np.uint8)

        # Add a mask if requested and return
        if mask_type != None:
            overlay = norm_data.copy()
            mask = self.masks[mask_type]['mask']
            label_map = self.masks[mask_type]['metadata']['label_map']
            color_map = self.masks[mask_type]['metadata']['color_map']
            for label, _ in label_map.items():
                if label != 0:
                    overlay[mask == label] = color_map[label]
            overlay = overlay.astype(np.uint8)
            return Image.fromarray(overlay)
        else:
            return Image.fromarray(norm_data)


    #Method to stack specified bands of the observation's items
    #RETURNS: Numpy Array and X Array
    def stack(self, bands, items = None, aoi = None):
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
        xx = xx[bands].median(dim="time")
        image_array = (
            xx
            .to_array()
            .transpose("y", "x", "variable")
            .values
        )
        return image_array, xx


    # Define the total cloud fraction of the observation
    def set_cloud_fraction(self):
        item_stac = self.stack(['SCL'])

        # Check for cloud cover
        scl = item_stac[0][:,:,0]
        cloud_mask = np.isin(scl, [1, 3, 8, 9, 10,11]).astype('int64')
        self.cloud_fraction = cloud_mask.mean()


    # Returns the entire tile image for analysis
    def get_whole_item(self, ind):
        signed_item = planetary_computer.sign(self.items[ind])
        visual_href = signed_item.assets["visual"].href
        img = rio.open_rasterio(visual_href)
        return img
    


    # Setup mgrs refefences
    def setup_mgrs_ref(self, mgrs_tiles):
        gdf = gpd.read_file(MGRS_REFERENCE_FILE)
        tile_col = "Name"

        subset = gdf[gdf[tile_col].isin(mgrs_tiles)].copy()
        subset = subset.set_index(tile_col)
        self.mgrs_gdf = subset
        del gdf
        gc.collect()



    # Method to set the NDVI values
    def set_ndvi(self):
        ndvi_bands = ['B04','B08']
        ndvi_stack = self.stack(ndvi_bands)
        ndvi_xx=ndvi_stack[1]
        red_band = ndvi_xx['B04']
        nir_band = ndvi_xx['B08']

        self.ndvi = (nir_band-red_band)/(nir_band+red_band)


    # Method to setup NDVI and all other indexes
    def set_all_indexes(self):
        bands = ['B04','B08', 'B11']
        xx = self.stack(bands)[1]
        red_band = xx['B04']
        nir_band = xx['B08']
        swir_band = xx['B11']

        self.ndvi = (nir_band-red_band)/(nir_band+red_band)
        self.fvc = ((self.ndvi-0.2)/(0.8-0.2))**2
        self.savi = ((nir_band-red_band)/(nir_band+red_band+0.5))*1.5
        self.ndmi = ((nir_band-swir_band)/(nir_band+swir_band))
        

    # Method to return NDVI Heatmap as a PIL Image
    def ndvi_heatmap_only(self, cmap="viridis"):
        ndvi_arr = self.ndvi.to_numpy()
        arr = np.clip(ndvi_arr, 0, 1)

        # Apply colormap from plt
        rgba = plt.get_cmap(cmap)(arr)

        # Convert to uint8 RGB
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(rgb)




