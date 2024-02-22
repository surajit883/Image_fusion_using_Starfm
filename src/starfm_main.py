#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to perform prediction using STARFM algorithm on Landsat and MODIS NDVI data.

Author: Surajit Hazra
"""


import ee
import numpy as np 
from datetime import datetime, timedelta

from scipy import interpolate
from starfm4py import create_dask_arr
import starfm4py as stp 

ee.Initialize()


def main(start_date, end_date, coord): 
    
    """
    Main function to perform prediction using STARFM algorithm.

    Parameters:
    start_date (str): Start date for data collection in 'YYYY-MM-DD' format.
    end_date (str): End date for data collection in 'YYYY-MM-DD' format.
    coord (list of list): List of coordinates defining the area of interest.
    """
    global clip_image, AOI,landsat_shape
    
    # Function to clip image to the area of interest
    def clip_image(image) :
        return image.clip(AOI)
       
    # Function to filter dates between Landsat and MODIS nearest date or same date
    def filter_dates(modis_dates, landsat_dates, max_difference_days=5):
        filter_dict = {}
    
        for idx, date1 in enumerate(modis_dates):
            diffs = [abs((date1 - date2).days) for date2 in landsat_dates]
    
            min_position = diffs.index(min(diffs))
            min_date = landsat_dates[min_position]
    
            if min(diffs) <= max_difference_days:
                if (date1 not in filter_dict.keys()) and (min_date not in filter_dict.values()) :
                    filter_dict[date1] = min_date
                    
                else:
                    prev_diff = abs((list(filter_dict.keys())[-1] - list(filter_dict.values())[-1]).days)
                    if prev_diff > min(diffs):
                        filter_dict.popitem()
                        filter_dict[date1] = min_date
    
        return filter_dict
    
    # Function to sample rectangle from image    
    def sample_rectangle(image):
        return image.sampleRectangle(AOI, defaultValue = -1)
    
    # Function to convert feature properties to arrays
    def get_arrays(feature) :
        arr = np.array(feature['properties']['NDVI']).astype(float)
        arr[arr < 0] = np.nan
        return arr
    
    # Function to fetch Landsat data
    def get_landsat_data(start_date, end_date, AOI) :
        """
        Fetch Landsat data for the specified time period and area of interest.

        Parameters:
        start_date (str): Start date for data collection in 'YYYY-MM-DD' format.
        end_date (str): End date for data collection in 'YYYY-MM-DD' format.
        AOI (ee.Geometry): Area of interest as Earth Engine Geometry.

        Returns:
        tuple: Tuple containing Landsat dates and feature arrays.
        """
        # Function to calculate NDVI
        def ndvi_func(image):
            red = image.select('SR_B4').multiply(0.0000275).add(-0.2)
            nir = image.select('SR_B5').multiply(0.0000275).add(-0.2)
            ndvi = nir.subtract(red).divide(nir.add(red))
            return image.addBands(ndvi.rename("NDVI"))
    
        # Function to mask clouds in Landsat images
        def landsat_cloud_mask(image):
            cloudShadowBitMask = (1 << 3)
            cloudsBitMask = (1 << 5)
            qa = image.select('QA_PIXEL')
            mask = (qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                    .And(qa.bitwiseAnd(cloudsBitMask).eq(0)))
            return image.updateMask(mask)
        
        # Function to sorted the images date wise 
        def get_sort_id(image) :
            id_ = ee.String(image.id()).split('_').get(-1)
            return image.set('sort_id', id_)       
            
        get_dates = np.vectorize(lambda x : x['id'].split('_')[-1])
        
        landsat_dataset = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                      .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                      .filterBounds(AOI)
                      .filterDate(start_date, end_date)
                      .map(landsat_cloud_mask)
                      .map(ndvi_func)
                      .select('NDVI')
                      .map(get_sort_id)
                      .sort('sort_id')
                      .map(clip_image)
                      
                     )
        landsat_dataset_v2 = landsat_dataset.map(sample_rectangle)
        landsat_features = landsat_dataset_v2.getInfo()['features']
        landsat_dates = get_dates(landsat_features)
        landsat_feature_arrays = np.array(list(map(get_arrays, landsat_features)))
        return landsat_dates, landsat_feature_arrays
    
    # Function to fetch MODIS data
    def get_modis_data(start_date, end_date, AOI):
        """
        Fetch MODIS data for the specified time period and area of interest.

        Parameters:
        start_date (str): Start date for data collection in 'YYYY-MM-DD' format.
        end_date (str): End date for data collection in 'YYYY-MM-DD' format.
        AOI (ee.Geometry): Area of interest as Earth Engine Geometry.

        Returns:
        tuple: Tuple containing MODIS dates and feature arrays.
        """

        # Function to scale NDVI

        def scale_factor(image) :
            ndvi = image.select('NDVI').multiply(0.0001)
            resampled_img = ndvi.resample('bicubic').reproject(crs = 'EPSG:4326', scale = 30)
            return resampled_img
        
        get_dates = np.vectorize(lambda x : x['id'].replace('_', ''))
        
        modis_dataset = (ee.ImageCollection("MODIS/061/MOD13Q1")
                         .filterBounds(AOI)
                         .filterDate(start_date, end_date)
                         .select('NDVI')
                         .map(scale_factor)
                         .map(clip_image)
                        )
        modis_dataset_v2 = modis_dataset.map(sample_rectangle)
        modis_features = modis_dataset_v2.getInfo()['features']
        modis_dates = get_dates(modis_features)
        modis_feature_arrays = np.array(list(map(get_arrays, modis_features)))
        return modis_dates, modis_feature_arrays
    
    # Define Area of Interest (AOI)
    AOI = ee.Geometry.Polygon(coord)

    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch Landsat data
    landsat_dates, landsat_feature_arrays = get_landsat_data(start_date, end_date, AOI)
    landsat_datetime = [datetime.strptime(date, '%Y%m%d') for date in landsat_dates]
    landsat_shape = landsat_feature_arrays[0].shape

    # Fetch Landsat data
    modis_dates, modis_feature_arrays = get_modis_data(start_date, end_date, AOI)
    modis_dates, modis_feature_arrays3 = get_modis_data(start_date, end_date, AOI)
    modis_datetime = [datetime.strptime(date, '%Y%m%d') for date in modis_dates]
    
    modis_shape = modis_feature_arrays[0].shape
    
    # Resample arrays if shapes are mismatched
    if landsat_shape != modis_shape :
        get_min = np.array([landsat_shape, modis_shape]).min(axis = 0)
        new_r = get_min[0]
        new_c = get_min[1]
    
        def resample_mismatched_array(array, new_r=new_r, new_c=new_c):
            xrange = lambda x: np.linspace(0,1,x)
            f = interpolate.interp2d(xrange(array.shape[1]), 
                                     xrange(array.shape[0]), 
                                     array, kind = "linear")
            return f(xrange(new_c), xrange(new_r))
        
        if (new_r, new_c) == landsat_shape :
            modis_feature_arrays_v2 = np.array(list(map(resample_mismatched_array, 
                                     modis_feature_arrays)))
            landsat_feature_arrays_v2 = landsat_feature_arrays
        else :
            landsat_feature_arrays_v2 = np.array(list(map(resample_mismatched_array, 
                                     landsat_feature_arrays)))
            modis_feature_arrays_v2 = modis_feature_arrays   


    # Filter dates between Landsat and MODIS
    filtered_dict = filter_dates(modis_datetime, landsat_datetime)
    fil_land_dt = [datetime.strftime(date, '%Y%m%d') for date in list(filtered_dict.values())]
    fil_mod_dt = [datetime.strftime(date, '%Y%m%d') for date in list(filtered_dict.keys())]
    
    # Get indices for filtered dates
    seen = set()
    land_indices = [i for i, val in enumerate(landsat_dates) 
                    if val in fil_land_dt and val not in seen and not seen.add(val)]
    mod_indices = [i for i, val in enumerate(modis_dates) if val in fil_mod_dt]
    
    # Get feature arrays for filtered dates
    landsat_arrs = landsat_feature_arrays_v2[land_indices]
    modis_arrs = modis_feature_arrays_v2[mod_indices]
    
    predicted_imgs = {}
    for i in range(len(fil_land_dt) - 1) :
        if ~np.isnan(landsat_arrs[i]).all() :
            # Create dask array from array
            land_pred_img = create_dask_arr(landsat_arrs[i], modis_arrs[i], modis_arrs[i + 1])
            predicted_imgs.update({fil_mod_dt[i + 1] :land_pred_img})
            
    
            img_shape = landsat_arrs[i].shape
            sizeSlices = img_shape[0]
            S2_t0 = create_dask_arr(landsat_arrs[i])
            S3_t0 = create_dask_arr(modis_arrs[i])
            S3_t1 = create_dask_arr(modis_arrs[i + 1])
           
            for i in range(0, landsat_arrs[i].size-sizeSlices*img_shape[1]+1, sizeSlices*img_shape[1]):
                fine_image_t0 = S2_t0[i:i+sizeSlices*img_shape[1],]
                coarse_image_t0 = S3_t0[i:i+sizeSlices*img_shape[1],]
                coarse_image_t1 = S3_t1[i:i+sizeSlices*img_shape[1],]
                
                # Perform prediction using STARFM
                prediction = stp.starfm(fine_image_t0, coarse_image_t0, coarse_image_t1, img_shape)
        
                if i == 0:
                    predictions = prediction
        
                else:
                    predictions = np.append(predictions, prediction, axis=0)
                  
    return predicted_imgs
if __name__ == "__main__":
    
    coords = [[80.05795776844025, 27.702334537452597],
     [80.05842078477144, 27.703888806777883],
     [80.05724027752876, 27.704179413501716],
     [80.05703777074814, 27.702377579968413],
     [80.05795776844025, 27.702334537452597]]
    
    start_date = '2024-01-01'
    end_date = '2024-01-20'
    
    # Call main function
    main(start_date, end_date, coords)