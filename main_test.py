from main import split_building_limits, rasterize_geodataframe, check_height_plateau_cover_building_limits,find_holes
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches


def test_split_building_limits():

    building_limits_gdf = gpd.GeoDataFrame({
        'name': ['building1'],
        'geometry': [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        ]
    })
    height_plateaus_gdf = gpd.GeoDataFrame({
        'name': ['plateau1', 'plateau2', 'plateau3'],
        'geometry': [
            Polygon([(-1, -1), (-1, 0), (0, 0), (0, -1)]),
            Polygon([(-1, -1), (1, 2), (2, 2), (2, 1)]),
            Polygon([(3, 3), (3, 4), (4, 4), (4, 3)])
        ]
    })
    result = split_building_limits(building_limits_gdf, height_plateaus_gdf)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    for _,row in result.iterrows():
        print(row)

    # define the vertices of the polygon
    vertices1 = [(0, 0), (1, 0), (1, 1), (0, 1)]
    vertices2 = [(-1, -1), (-1, 0), (0, 0), (0, -1)]
    vertices3 = [(-1, -1), (1, 2), (2, 2), (2, 1)]
    vertices4 = [(3, 3), (3, 4), (4, 4), (4, 3)]

    # create a Polygon object from the vertices
    polygon1 = patches.Polygon(vertices1, alpha=0.5)
    polygon2 = patches.Polygon(vertices2, alpha=0.5)
    polygon3 = patches.Polygon(vertices3, alpha=0.5)
    polygon4 = patches.Polygon(vertices4, alpha=0.5)

    # create a plot and add the polygon to it
    fig, ax = plt.subplots()
    ax.add_patch(polygon1)
    ax.add_patch(polygon2)
    ax.add_patch(polygon3)
    ax.add_patch(polygon4)

    # set the x and y limits
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    # show the plot
    #plt.show()


def test_rasterize_geodataframe_with_null_geometry():
    # Create a GeoDataFrame with one valid polygon and one null geometry
    gdf = gpd.GeoDataFrame(
        {
            'name': ['polygon1'],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            ]
        }
    )

    # Call the rasterize_geodataframe() function and expect it to return a dictionary with one entry
    # and the correct bounding box and resolution values
    rasters, xmin, ymin, xmax, ymax, xres, yres = rasterize_geodataframe(gdf)

    assert len(rasters) == 1
    assert '0' in rasters
    assert isinstance(rasters['0'], np.ndarray)
    assert xmin == 0
    assert ymin == 0
    assert xmax == 1
    assert ymax == 1
    assert xres == 0.0001
    assert yres == 0.0001

def test_check_height_plateau_cover_building_limits():
    # Test case 1
    rasters = {"0.0": np.array([[1,1,1],[1,1,1],[1,1,1]]),
               "1.0": np.array([[1,1,1],[1,1,1],[1,1,1]]),
               "2.0": np.array([[1,1,1],[1,1,1],[1,1,1]])}
    assert check_height_plateau_cover_building_limits(rasters) == True

    # Test case 2
    rasters = {"0.0": np.array([[1,1,1],[1,1,1],[1,1,1]]),
               "1.0": np.array([[0,1,1],[1,1,1],[1,1,1]]),
               "2.0": np.array([[0,1,1],[1,1,1],[1,1,1]])}
    assert check_height_plateau_cover_building_limits(rasters) == False

    # Test case 4
    rasters = {"0.0": np.array([[1,1,1],[1,1,1],[1,1,1]]),
               "1.0": np.array([[1,1,1],[1,1,1],[1,1,0]]),
               "2.0": np.array([[1,1,1],[1,1,1],[1,1,0]])}
    assert check_height_plateau_cover_building_limits(rasters) == False


def test_find_holes():
    # Test case 1
    rasters = {"0.0": np.array([[1,1,1],[1,1,1],[1,1,1]]),
               "1.0": np.array([[1,1,1],[0,0,0],[0,0,0]]),
               "2.0": np.array([[1,1,1],[1,1,1],[1,1,1]])}
    assert find_holes(rasters) == False

    print("----")

    # Test case 2
    rasters = {"0.0": np.array([[1,1,1],[1,1,1],[1,1,1]]),
               "1.0": np.array([[1,1,1],[1,0,0],[0,0,0]]),
               "2.0": np.array([[0,0,0],[0,0,1],[1,1,1]])}
    assert find_holes(rasters) == True

    print("----")


    # Test case 4
    rasters = {"0.0": np.array([[1,1,1],[1,1,1],[1,1,1]]),
               "1.0": np.array([[0,0,0],[0,0,0],[1,1,0]]),
               "2.0": np.array([[1,1,1],[1,1,1],[0,0,0]])}
    assert find_holes(rasters) == False