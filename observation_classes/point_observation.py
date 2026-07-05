# Custom Functions
from utils.helper import point_to_bbox, crop32
from models.utils.display import sentinel_worldcover_image_and_mask_display as wc_display

# Basic Libraries
import base64
import warnings
import odc.stac
import webcolors
import numpy as np
from math import sqrt
from PIL import Image
from io import BytesIO
import rioxarray as rio
import planetary_computer
from math import log as ln
from shapely import from_wkt
import matplotlib.pyplot as plt
from pystac_client import Client
from shapely.geometry import shape
from shapely.ops import unary_union
from pympler.asizeof import asizeof
from datetime import datetime, timedelta

# Retry libraries
from urllib3 import Retry
from pystac_client.stac_api_io import StacApiIO


PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"



def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    header = "data:image/png;base64,"
    return header + str(base64.b64encode(buffer.getvalue()).decode("utf-8"))

def closest_css3_color(rgb):
    r, g, b = rgb
    min_distance = float("inf")
    closest_name = None
    for name in webcolors.names("css3"):
        cr, cg, cb = webcolors.hex_to_rgb(
            webcolors.name_to_hex(name)
        )
        distance = sqrt(
            (r - cr) ** 2 +
            (g - cg) ** 2 +
            (b - cb) ** 2
        )
        if distance < min_distance:
            min_distance = distance
            closest_name = name

    return closest_name


#SPECIFICALLY FOR SENTINEL ITEMS
#Class to collect items for a spacific observation area at a given time
class ObservedArea:
    def __init__(
            self, 
            lat,
            lon,
            sqkm, 
            collection: list,
            date_window: str, 
            cloud_cover = 10
            ):

        # ---------------------------
        # Setup the Client
        # ---------------------------
        retry = Retry(total=5, backoff_factor=1,status_forcelist=[502, 503, 504],allowed_methods=None)
        stac_api_io = StacApiIO(max_retries=retry)
        catalog = Client.open(PLANETARY_COMPUTER_URL, stac_io = stac_api_io)


        self.aoi = point_to_bbox(lat, lon, sqkm)
        self.date_window = date_window
        self.collection = collection
        self.catalog = catalog
        self.items = []
        self.date = None
        self.masks = {}
        self.batch = None
        self.sentinel = (self.collection ==["sentinel-2-l2a"])


        # Get the items
        self.get_items(date_window, cloud_cover)




    #Function to collect the items associated with that area during that time
    def get_items(self, date_window, cloud_cover):

        #Selects only the most recent item from each MGRS tile
        def filter_items_sentinel(items):
            #filter out those with clouds over AOI
            cloudless_items = []
            for item in items:
                scl = self.stack(['SCL'],[item])[0]
                cloud_mask = np.isin(scl, [1, 3, 7, 8, 9, 10,11]).astype('int64')
                cloud_fraction = cloud_mask.mean()
                if cloud_fraction < 0.1: cloudless_items.append(item)
            
            #Only take the most recent item of each MGRS grid
            latest_items = {}
            for item in cloudless_items:
                # Get the Grid
                tile_id = item.properties.get("s2:mgrs_tile")
                if tile_id and tile_id not in latest_items:
                    latest_items[tile_id] = item
            return list(latest_items.values())
        
        def filter_items_landsat(items):
            cloudless_items = []
            for item in items:
                ds = self.stack(['qa_pixel'],[item])
                qa = ds["qa_pixel"]
                cloud_mask = (qa.astype("uint16") & (1 << 3)) > 0
                cloud_fraction = cloud_mask.mean(dim=["x","y"])
                cloud_fraction = cloud_fraction.compute().item()

                if cloud_fraction < 0.1: cloudless_items.append(item)    

            #Only take the most recent item of each row/path
            def landsat_tile_id(item):
                path = int(item.properties.get("landsat:wrs_path"))
                row = int(item.properties.get("landsat:wrs_row"))
                return f"{path:03d}_{row:03d}"
            
            latest_items = {}
            for item in cloudless_items:
                # Get the Grid
                tile_id = landsat_tile_id(item)
                if tile_id and tile_id not in latest_items:
                    latest_items[tile_id] = item
            return list(latest_items.values())         





        
        #Confirms if the items cover the observation AOI
        def confirm_coverage(items):
            item_geoms = [shape(item.geometry) for item in items]
            combined_geom = unary_union(item_geoms)
            intersection = combined_geom.intersection(self.aoi)
            self.coverage = intersection.area / self.aoi.area

            if self.coverage > 0.9: return True
            else: return False

        # Sets the date for the observation
        # If observation contains items from multiple dates, will select the oldest date (first date)
        def set_date():
            dates = []
            for item in self.items:
                date_str = item.properties['datetime'][:10]
                dates.append(datetime.strptime(date_str,'%Y-%m-%d'))
            self.date = min(dates).date()       

        #---------------------------------------------
        #Search
        #---------------------------------------------
        warnings.filterwarnings("ignore")
        search = self.catalog.search(
            collections=self.collection,
            bbox = self.aoi.bounds,
            datetime=date_window,
            query={"eo:cloud_cover": {"lt": cloud_cover}},
            sortby="eo:cloud_cover",
            max_items = 10
        )
        items = search.get_all_items()
        if self.sentinel: items = filter_items_sentinel(items)
        else: items = filter_items_landsat(items)

        if len(items) >= 0 and confirm_coverage(items): 
            self.items = items
            set_date()
            print(f'Observation collected on {self.date}')
            return True
        else: raise ValueError('No clear observation could be found of this area in this date range')



#---------------------------------------------------------------------
#-----------------------Support-Methods-------------------------------
#---------------------------------------------------------------------


    #Method to stack specified bands of the observation's items
    #RETURNS: Numpy Array and X Array
    def stack(self, bands, items = None):
        if items == None: items = self.items
        #Sign the items
        signed_items = []
        for item in items: signed_items.append(planetary_computer.sign(item))
        
        #collect the xarray
        xx = odc.stac.load(
            signed_items,
            bands = bands,
            geopolygon=self.aoi,
            resampling = 'bilinear',
            chunks = {'x': 512, 'y': 512}
        )
        
        if self.sentinel: 
            xx = xx[bands].median(dim="time")
            image_array = (
                xx
                .to_array()
                .transpose("y", "x", "variable")
                .values
            )
            return image_array, xx

        else: return xx[bands].astype("uint16")



        
    


    #Method to quickly return the visual as a PIL image
    def get_image(self, mask_type = None, target_sat = 75):
        if self.collection != ["sentinel-2-l2a"]:
            print('Obserservation has no image data and cannot return an image')
            return

        data = self.stack(['B02','B03','B04'])[0]
        data =crop32(np.transpose(data,(2,0,1)))
        data = np.transpose(data,(1,2,0))
        norm_data = np.zeros(data.shape)


        for i in range(data.shape[2]):
            band = data[:,:,i]
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



    # Returns the entire tile image for analysis
    def get_whole_item(self, ind):
        if self.collection != ["sentinel-2-l2a"]:
            print('Obserservation has no image data and cannot return an image')
            return
        signed_item = planetary_computer.sign(self.items[ind])
        visual_href = signed_item.assets["visual"].href
        img = rio.open_rasterio(visual_href)
        return img



    def pack(self):
        date_format = "%Y-%m-%d %H:%M:%S"
        return {
            'aoi': self.aoi.wkt,
            'collection': self.collection,
            'coverage': float(self.coverage),
            'date': self.date.strftime(date_format),
            'target_date': self.target_date.strftime(date_format),
            'item_ids': [item.id for item in self.items]
        }
     
    @classmethod
    def unpack(cls, data):
 
        obs = cls.__new__(cls)
        date_format = "%Y-%m-%d %H:%M:%S"

        #Unpack the basics
        obs.aoi = from_wkt(data['aoi'])
        retry = Retry(total=5, backoff_factor=1,status_forcelist=[502, 503, 504],allowed_methods=None)
        stac_api_io = StacApiIO(max_retries=retry)
        obs.catalog = Client.open(PLANETARY_COMPUTER_URL, stac_io = stac_api_io)
        obs.target_date = datetime.strptime(data['target_date'], date_format)

        obs.collection = data['collection']
        obs.coverage = data['coverage']
        obs.date = datetime.strptime(data['date'], date_format)
        obs.masks = []



        #Collect the items
        srch = obs.catalog.search(collections=obs.collection, ids=data['item_ids'])
        signed_items = [planetary_computer.sign(item) for item in srch.get_items()]
        obs.items = signed_items

        return obs
        



