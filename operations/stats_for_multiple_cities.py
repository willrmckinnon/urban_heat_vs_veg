
from utils.calculate_obs_stats import calc_iou_thresholds
from utils.calculate_obs_stats import calculate_index_stats

def best_iou_thresholds(obs):
    output = {}
    ndvi_row = calc_iou_thresholds(obs, obs.ndvi)
    savi_row = calc_iou_thresholds(obs, obs.savi)
    fvc_row = calc_iou_thresholds(obs, obs.fvc)
    ndmi_row = calc_iou_thresholds(obs, obs.ndmi)

    output['ndvi_lst_thesh'] = ndvi_row['lst_threshold'].values[0]
    output['ndvi_thesh'] = ndvi_row['index_threshold'].values[0]
    output['savi_lst_thesh'] = savi_row['lst_threshold'].values[0]
    output['savi_thesh'] = savi_row['index_threshold'].values[0]
    output['fvc_lst_thesh'] = fvc_row['lst_threshold'].values[0]
    output['fvc_thesh'] = fvc_row['index_threshold'].values[0]
    output['ndmi_lst_thesh'] = ndmi_row['lst_threshold'].values[0]
    output['ndmi_thesh'] = ndmi_row['index_threshold'].values[0]

    return output



def complete_statistical_analysis(obs,
                                  ndvi_thresh = 0.5,
                                  ndvi_lst_thresh = 0.34,
                                  savi_thresh = 0.504,
                                  savi_lst_thresh = 0.5,
                                  fvc_thresh = 0.1,
                                  fvc_lst_thresh = 0.5,    
                                  ndmi_thresh = 0.09,
                                  ndmi_lst_thresh = 0.5,                              
                                  ):
    """
    Receives an observation whose lst and index data has been refreshed
    returns three dictionaries that are meant to be equivalent to three entries for pandas df's
    First return: complete dictionary with all statistical data
    Second return: dictionary with raw index pixel statistics
    Third return: dictionary with mean index pixel statistics where each pixel represents mean index values for 150m in any direction
    """ 

    output_complete = {'city_name': obs.city_name.iloc[0]}
    output_raw = {'city_name': obs.city_name.iloc[0]}
    output_mean = {'city_name': obs.city_name.iloc[0]}


    # NDVI
    raw_ndvi_stats = calculate_index_stats(obs, 
                              obs.ndvi, 
                              return_figure=False,
                              iou_index_thresh=ndvi_thresh,
                              iou_lst_thresh=ndvi_lst_thresh
                              )
    new_raw_data = {
        'raw_ndvi_p-value': raw_ndvi_stats['p-value'],
        'raw_ndvi_pearson_cor': raw_ndvi_stats['pearson_cor'],
        'raw_ndvi_regression_slope': raw_ndvi_stats['linear_regression_slope'],
        'raw_ndvi_iou': raw_ndvi_stats['iou'],
    }
    output_raw |= new_raw_data

    mean_ndvi_stats = calculate_index_stats(obs, 
                              obs.ndvi, 
                              return_figure=False,
                              iou_index_thresh=ndvi_thresh,
                              iou_lst_thresh=ndvi_lst_thresh,
                              run_local_mean=True
                              )
    new_mean_data = {
        'mean_ndvi_p-value': mean_ndvi_stats['p-value'],
        'mean_ndvi_pearson_cor': mean_ndvi_stats['pearson_cor'],
        'mean_ndvi_regression_slope': mean_ndvi_stats['linear_regression_slope'],
        'mean_ndvi_iou': mean_ndvi_stats['iou'],
    }
    output_mean |= new_mean_data

    output_complete['scene_mean_lst'] = raw_ndvi_stats['scene_mean_lst']
    output_complete['scene_mean_ndvi'] = raw_ndvi_stats['scene_mean_index']







    # SAVI
    raw_savi_stats = calculate_index_stats(obs, 
                              obs.savi, 
                              return_figure=False,
                              iou_index_thresh=savi_thresh,
                              iou_lst_thresh=savi_lst_thresh
                              )
    new_raw_data = {
        'raw_savi_p-value': raw_savi_stats['p-value'],
        'raw_savi_pearson_cor': raw_savi_stats['pearson_cor'],
        'raw_savi_regression_slope': raw_savi_stats['linear_regression_slope'],
        'raw_savi_iou': raw_savi_stats['iou'],
    }
    output_raw |= new_raw_data


    mean_savi_stats = calculate_index_stats(obs, 
                              obs.savi, 
                              return_figure=False,
                              iou_index_thresh=savi_thresh,
                              iou_lst_thresh=savi_lst_thresh,
                              run_local_mean=True
                              )
    new_mean_data = {
        'mean_savi_p-value': mean_savi_stats['p-value'],
        'mean_savi_pearson_cor': mean_savi_stats['pearson_cor'],
        'mean_savi_regression_slope': mean_savi_stats['linear_regression_slope'],
        'mean_savi_iou': mean_savi_stats['iou'],
    }
    output_mean |= new_mean_data

    output_complete['scene_mean_savi'] = raw_savi_stats['scene_mean_index']




    # FVC
    raw_fvc_stats = calculate_index_stats(obs, 
                              obs.fvc, 
                              return_figure=False,
                              iou_index_thresh=fvc_thresh,
                              iou_lst_thresh=fvc_lst_thresh
                              )
    new_raw_data = {
        'raw_fvc_p-value': raw_fvc_stats['p-value'],
        'raw_fvc_pearson_cor': raw_fvc_stats['pearson_cor'],
        'raw_fvc_regression_slope': raw_fvc_stats['linear_regression_slope'],
        'raw_fvc_iou': raw_fvc_stats['iou'],
    }
    output_raw |= new_raw_data
    

    mean_fvc_stats = calculate_index_stats(obs, 
                              obs.fvc, 
                              return_figure=False,
                              iou_index_thresh=fvc_thresh,
                              iou_lst_thresh=fvc_lst_thresh,
                              run_local_mean=True
                              )
    new_mean_data = {
        'mean_fvc_p-value': mean_fvc_stats['p-value'],
        'mean_fvc_pearson_cor': mean_fvc_stats['pearson_cor'],
        'mean_fvc_regression_slope': mean_fvc_stats['linear_regression_slope'],
        'mean_fvc_iou': mean_fvc_stats['iou'],
    }
    output_mean |= new_mean_data

    output_complete['scene_mean_fvc'] = raw_fvc_stats['scene_mean_index']





    # NDVI
    raw_ndmi_stats = calculate_index_stats(obs, 
                              obs.ndmi, 
                              return_figure=False,
                              iou_index_thresh=ndmi_thresh,
                              iou_lst_thresh=ndmi_lst_thresh
                              )
    new_raw_data = {
        'raw_ndmi_p-value': raw_ndmi_stats['p-value'],
        'raw_ndmi_pearson_cor': raw_ndmi_stats['pearson_cor'],
        'raw_ndmi_regression_slope': raw_ndmi_stats['linear_regression_slope'],
        'raw_ndmi_iou': raw_ndmi_stats['iou'],
    }
    output_raw |= new_raw_data


    mean_ndmi_stats = calculate_index_stats(obs, 
                              obs.ndmi, 
                              return_figure=False,
                              iou_index_thresh=ndmi_thresh,
                              iou_lst_thresh=ndmi_lst_thresh,
                              run_local_mean=True
                              )
    new_mean_data = {
        'mean_ndmi_p-value': mean_ndmi_stats['p-value'],
        'mean_ndmi_pearson_cor': mean_ndmi_stats['pearson_cor'],
        'mean_ndmi_regression_slope': mean_ndmi_stats['linear_regression_slope'],
        'mean_ndmi_iou': mean_ndmi_stats['iou'],
    }
    output_mean |= new_mean_data

    output_complete['scene_mean_ndmi'] = raw_ndmi_stats['scene_mean_index']    


    output_complete |= output_raw
    output_complete |= output_mean

    return output_complete, output_raw, output_mean



