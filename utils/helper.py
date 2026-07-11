# Basic Libraries
import yaml
import numpy as np
from PIL import Image
from shapely.ops import transform
from pyproj import CRS,Transformer
from PIL import ImageDraw, ImageFont
from rasterio.transform import Affine
from shapely.geometry import box, Polygon






def create_overlay(
    base_image: Image.Image,
    arr: np.ndarray,
    arr_threshold=0.4,
    arr_color=(255, 0, 0),
):
    """
    Overlay an arrays onto a PIL image.
    ----------
    base_image : PIL.Image
        Original image.
    arr : np.ndarray
        Values between 0 and 1.
    """
    def resize_array(arr: np.ndarray, target_shape):
        target_h, target_w = target_shape
        arr_img = Image.fromarray(arr.astype(np.float32))
        arr_resized = arr_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(arr_resized)


    # --------------------------------------------------
    # SIZE ARRAY
    # --------------------------------------------------

    base = base_image.convert("RGBA")
    img_w, img_h = base.size
    
    arr = resize_array(arr, (img_h, img_w))


    # --------------------------------------------------
    # ARR OVERLAY
    # --------------------------------------------------

    arr_alpha = np.zeros_like(arr, dtype=np.uint8)

    mask = arr > arr_threshold

    if mask.any():
        # Normalize values above threshold to 0-255
        arr_norm = (arr[mask] - arr_threshold) / (
            np.nanmax(arr) - arr_threshold
        )

        arr_alpha[mask] = (arr_norm * 255).astype(np.uint8)

    overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    overlay[..., 0] = arr_color[0]
    overlay[..., 1] = arr_color[1]
    overlay[..., 2] = arr_color[2]
    overlay[..., 3] = arr_alpha

    overlay1_img = Image.fromarray(overlay, mode="RGBA")


    # --------------------------------------------------
    # COMPOSITE
    # --------------------------------------------------

    return Image.alpha_composite(base, overlay1_img)








def add_legend(img, indices, text_size = 20, ind_length = 220):

    legend_overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(legend_overlay)

    h, w = legend_overlay.size
    # Legend position
    x = w-30
    y = h-30
    box_size = 20
    spacing = 35



    try:
        font = ImageFont.truetype("Arial.ttf", text_size)
    except:
        font = ImageFont.load_default()

    # Draw semi-transparent background
    draw.rounded_rectangle([x-(len(indices)*ind_length+10), y-text_size-10, x, y+10], radius=10, fill=(0, 0, 0, 125))

    for name, clr in indices.items():
        i = list(indices).index(name)+1
        x_offset = i*ind_length

        draw.rectangle([x-x_offset, y-box_size, x-x_offset+box_size, y], fill=clr, outline="white", width=1)
        draw.text((x-x_offset+box_size+10, y-text_size), name, fill="white", font=font)
    
    
    return Image.alpha_composite(img, legend_overlay)









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
