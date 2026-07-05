# Import Generic Libraries
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from scipy.stats import pearsonr
from scipy.stats import linregress
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter




# Returns numpy arrays for lst and ndvi
def numpy_masks(obs, index):
    #Resample to pair up the lst and ndvi
    ndvi_10m_xx = index
    lst_xx = obs.lst

    ndvi_xx = ndvi_10m_xx.rio.reproject_match(lst_xx, resampling=Resampling.average)

    # Convert to numpy 
    ndvi = ndvi_xx.to_numpy().squeeze()
    lst = lst_xx.to_array().to_numpy().squeeze()
    return ndvi, lst    


# Returns a xarray the same size as input where each value is the mean of all surrounding it
# radius is number of pixels in each direction to go
def local_mean(da, radius):
    window = 2 * radius + 1
    return da.rolling(x=window, y=window,center=True).mean()



# Returns a pair of matching pixel values as numpy arrays
def mask_pairing(index_s, lst_s):
    #Resample to pair up the lst and ndvi
    index_10m_xx = index_s
    lst_xx = lst_s

    index_xx = index_10m_xx.rio.reproject_match(lst_xx, resampling=Resampling.average)

    # Convert to numpy 
    index = index_xx.to_numpy().squeeze()
    lst = lst_xx.to_array().to_numpy().squeeze()

    # Flatten for comparison
    index_flat = index.ravel()
    lst_flat = lst.ravel()

    # Remove NaN
    mask = (np.isfinite(index_flat)& np.isfinite(lst_flat))
    index_valid = index_flat[mask]
    lst_valid = lst_flat[mask]

    return index_valid, lst_valid



# Calculate the best performing IOU thresholds
def calc_iou_thresholds(obs, index, return_all = False):
    ndvi, lst = numpy_masks(obs, index)
    rows = []
    for lst_thresh in np.arange(0.5, 1.55, 0.05):
        z_lst = (lst - np.nanmean(lst)) / np.nanstd(lst)
        lst_mask = z_lst > lst_thresh

        for ndvi_thresh in np.arange(0, 1.05, 0.05):
            ndvi_mask = ndvi < ndvi_thresh
            iou = np.sum(lst_mask & ndvi_mask) / np.sum(lst_mask | ndvi_mask)
            rows.append({
                'index_threshold': ndvi_thresh,
                'lst_threshold': lst_thresh,
                'IOU': iou
            })

    df = pd.DataFrame(rows)

    if return_all: return df
    else: return df[df['IOU'] == df['IOU'].max()]




def calculate_index_stats(obs,
                          index, 
                         return_figure=True,
                         iou_lst_thresh = 0.5,
                         iou_index_thresh = 0.35,
                         index_name = 'Index',
                         local_mean = False
                         ):
    
    if local_mean: index = local_mean(index)

    index_valid, lst_valid = mask_pairing(index, obs.lst)

    #################################
    #          Basic Stats          #
    #################################

    # Pearson Correlation
    r, p = pearsonr(index_valid, lst_valid)

    result = linregress(index_valid, lst_valid)
    linear_regression_slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue

    # Setup the return
    stats = {
        'scene_mean_index': index_valid.mean(),
        'scene_mean_lst': lst_valid.mean(),
        'p-value': p,
        'pearson_cor': r,
        'linear_regression_slope': linear_regression_slope
    }


    # Handle the figure generation
    if return_figure:
        # Generate regression line
        x_line = np.linspace(index_valid.min(), index_valid.max(),100)
        y_line = linear_regression_slope * x_line + intercept

        # Setup the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(index_valid, lst_valid, s=1, alpha=0.4, label='Correlation per Pixel')
        ax.plot(x_line, y_line, linewidth=2, alpha=0.3, color = 'red', label=f"Linear Fit (r={r_value:.2f})")

        ax.set_title(f'{obs.city_name.iloc[0]} LST v. {index_name}')
        ax.set_ylabel('Land Surface Temperature (LST) \u00b0C')
        ax.set_xlabel(f'{index_name}')
        ax.legend()

        # Save the figure as a pil image
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
        buffer.seek(0)

        # Convert to PIL
        fig_img = Image.open(buffer)
        plt.close(fig)

        stats['fig'] = fig_img



    #################################
    #           IOU Stats           #
    #################################

    lst_thresh = iou_lst_thresh # Standard Deviation offset to consider a heat anomoly
    index_thresh = iou_index_thresh # NDVI Threshold to consider it a vegetated area

    z_lst = (lst_valid - np.nanmean(lst_valid)) / np.nanstd(lst_valid)
    lst_mask = z_lst > lst_thresh
    index_mask = index_valid < index_thresh
    stats['iou'] = np.sum(lst_mask & index_mask) / np.sum(lst_mask | index_mask)

    return stats



