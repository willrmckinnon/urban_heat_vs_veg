# Custom Functions
from utils.helper import point_to_bbox

# Basic Libraries
import base64
import warnings
import odc.stac
import webcolors
from math import sqrt
from io import BytesIO
import planetary_computer
from shapely import from_wkt
from datetime import datetime
from pystac_client import Client
from shapely.geometry import shape
from shapely.ops import unary_union


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

        # Get the items
        self.get_items(date_window, cloud_cover)




    #Function to collect the items associated with that area during that time
    def get_items(self, date_window, cloud_cover):
        
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
        items = self.filter_items(items)

        
        if len(items) > 0: 
            if self.confirm_coverage(items):
                self.items = items
                set_date()
                print(f'Observation collected on {self.date} for {self.collection[0]}')
                return True
            else: raise ValueError('Items collected but did not cover the Area of Observation')
        else: raise ValueError('No clear observation could be found of this area in this date range')


    #Confirms if the items cover the observation AOI
    def confirm_coverage(self, items):
        item_geoms = [shape(item.geometry) for item in items]
        combined_geom = unary_union(item_geoms)
        intersection = combined_geom.intersection(self.aoi)
        self.coverage = intersection.area / self.aoi.area

        if self.coverage > 0.9: return True
        else: return False


    def filter_items(self, items):
        # Generic Placeholder function to be overwritten for specific
        # collection systems (e.g. Sentinel vs. Landsat)
        return items


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
        return xx[bands]
        





#---------------------------------------------------------------------
#-----------------------Saving-&-Reinitializing-----------------------
#---------------------------------------------------------------------


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
        



