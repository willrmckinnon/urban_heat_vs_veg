# Basic Libraries
import yaml
import numpy as np
from PIL import Image
from shapely.ops import transform
from pyproj import CRS,Transformer
from rasterio.transform import Affine
from shapely.geometry import box, Polygon


#Reads the config file
def load_config():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

#converts a point to a bbox
def point_to_bbox(lat, lon, sqkm):
    # Set the correct utm based on northern or southern lon
    utm_zone = int((lon + 180) / 6) + 1
    epsg = (32600 + utm_zone  if lat >= 0 else 32700 + utm_zone)

    # Set the CRS
    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(epsg)

    # Set the Transforms
    to_utm = Transformer.from_crs(wgs84, utm, always_xy=True)
    to_wgs = Transformer.from_crs(utm, wgs84, always_xy=True)

    x, y = to_utm.transform(lon, lat)
    target_width = 1000 * ((sqkm/0.9540802499563914)**(0.5))
    half_size = target_width/2

    square = box(x-half_size, y-half_size, x+half_size, y+half_size)
    bbox = transform(to_wgs.transform, square)

    return bbox

#converts a point to a polygon
def point_to_polygon(lat, lon, dim=4000):
    # WGS84 → Web Mercator (meters)
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_wgs = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # convert center point to meters
    x, y = to_m.transform(lon, lat)

    # create square in meters
    half_size = dim/2
    square = box(
        x - half_size, y - half_size,
        x + half_size, y + half_size
    )

    # convert back to WGS84
    coords = [
        list(to_wgs.transform(px, py))
        for px, py in square.exterior.coords
    ]

    return Polygon(coords)


#Function to convert np array into a PIL Image
def npy_to_img(img, saturation = 1):
    rgb = img[:,:,[2, 1, 0]].astype(np.float32)

    low = np.percentile(rgb,2)
    high = np.percentile(rgb, 98)

    rgb = (rgb-low)/(high - low)

    #Saturate and normalize
    sat = np.clip(rgb * saturation, 0, 1)
    norm_sat = (sat * 255).astype(np.uint8)

    
    return Image.fromarray(norm_sat)


#Crops input data to the nearest multiple of 32 for model handling
#INPUT: requires a (BxHxW) shapped numpy array
def crop32(data, transform = None):
    h_rem = data.shape[1] % 32
    w_rem = data.shape[2] % 32
    h = data.shape[1] - h_rem
    w = data.shape[2] - w_rem
    h_s = round(h_rem/2)
    w_s = round(w_rem/2)

    cropped_data = data[:, h_s:h_s+h, w_s:w_s+w]

    if transform: 
        cropped_transform = (transform * Affine.translation(w_s, h_s))
        return cropped_data, cropped_transform
    else:
        return cropped_data
