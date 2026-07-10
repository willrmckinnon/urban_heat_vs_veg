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




# Returns a pair of matching pixel values as numpy arrays
def mask_pairing(index_s, lst_s, obs, remove_water = True):
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

    # Remove Water
    if remove_water:
        scl_xx = obs.sentinel_obs.stack(['SCL'])[1]
        scl_r_xx = scl_xx.rio.reproject_match(lst_xx, resampling=Resampling.average)
        scl_arr = scl_r_xx.to_array().to_numpy().squeeze() 
        scl_flat = scl_arr.ravel()
        scl_valid = scl_flat[mask]

        # Water Mask where non water is true
        w_mask = scl_valid != 6 

        index_valid = index_valid[w_mask]
        lst_valid = lst_valid[w_mask]
        

    return index_valid, lst_valid





def calculate_index_stats(obs,
                          index, 
                          return_figure=True,
                          iou_lst_thresh = 0.5,
                          iou_index_thresh = 0.35,
                          index_name = 'Index',
                          run_local_mean = False,
                          remove_water = True
                          ):
    
    if index_name == 'NDVI': iou_index_thresh = 0.34
    elif index_name =='SAVI': iou_index_thresh = 0.504
    elif index_name == 'FVC': iou_index_thresh = 0.1
    elif index_name == 'NDMI': iou_index_thresh = 0.09
    else: iou_index_thresh = 0.3


    if run_local_mean: 
        window = 10
        index = index.rolling(x=window, y=window,center=True).mean()

    index_valid, lst_valid = mask_pairing(index, obs.lst, obs, remove_water=remove_water)

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











# Creates a histogram of temperature values
def temp_hist(obs):
    _, lst_valid = mask_pairing(obs.ndvi, obs.lst)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(lst_valid, bins=1000)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Frequency')
    ax.set_title('City Temperatures')

    # Save the figure as a pil image
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
    buffer.seek(0)

    # Convert to PIL
    fig_img = Image.open(buffer)
    plt.close(fig)
    return fig_img




# Calculate the best performing IOU thresholds
def calc_iou_thresholds(obs, index, return_all = False):
    #Resample to pair up the lst and ndvi
    ndvi_10m_xx = index
    lst_xx = obs.lst

    ndvi_xx = ndvi_10m_xx.rio.reproject_match(lst_xx, resampling=Resampling.average)

    # Convert to numpy 
    ndvi = ndvi_xx.to_numpy().squeeze()
    lst = lst_xx.to_array().to_numpy().squeeze()

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