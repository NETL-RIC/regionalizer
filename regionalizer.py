#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# regionalizer.py

#
# REQUIRED MODULES
#
import logging
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import requests


#
# DOCUMENTATION
#
__doc__ = """A universal spatial translation method from any
given spatial extent to the U.S. census tract level. This method is
designed for disaggregating numeric attributes from a larger spatial
scale (e.g., state level) down to the census tract or county;
it was not tested for upscaling.

In the current version, the starting spatial extents are predefined,
but any extent with an appropriate spatial data file
(e.g., shapefile or geoJSON) can be added to the code
(with appropriate methods to handle the data acquisition and
any necessary preprocessing, such as removing or renaming features).

The disaggregation methods are also predefined.
In the current version, two methods are available:
equal weighting (all census tracts within the boundary of the larger
spatial extent receive the same amount, 1/N), and
areal weighting (each census tract receives a portion from the larger
spatial extents in proportion to the fraction of coverage of each
feature in the larger spatial extent that a census tract covers).

Examples
--------
>>> from regionalizer import get_logger
>>> from regionalizer import get_m
>>> log = get_logger()
>>> M, codes, cen_codes = get_m(
...     starting_extent='ST',
...     ending_extent='CT',
...     weighting='Eq',
...     state='US'
... )

Notes
-----
The 'M' matrix is 2-D numpy array with rows equal to the number of starting
regions (e.g., 71 balancing authorities) and columns equal to the number of
ending regions (e.g., 84,122 for census tracts).

A list or 1-by-nrows array, representing an attribute at the starting region's
extent can be matrix-multiplied by 'M' to get an ncols-by-1 array of the same
attribute regionalized to census tract (or county).

Use the 'codes' (starting extent feature names) and 'cen_codes' (ending extent
feature names) lists to correctly map/label the values for further use.

Last updated:
    2025-04-11
"""
__all__ = [
    "area_weighting",
    "calculate_m",
    "calculate_water_scarcity_footprint",
    "correct_ba_geo_names",
    "correct_cb_geo_names",
    "correct_nb_geo_names",
    "correct_ns_geo_names",
    "download_file",
    "equal_weighting",
    "get_ba_geo",
    "get_ba_map",
    "get_cb_geo",
    "get_census_geo",
    "get_logger",
    "get_m",
    "get_nb_geo",
    "get_nercsub_geo",
    "map_ba_codes",
    "print_progress",
    "read_ba_codes",
    "read_m",
    "save_m",
    "tract_to_county",
    "tract_to_state",
    "DATA_DIR",
    "PCS",
    "ROPTS",
    "STATE_FILTER",
    "WOPTS",
]


#
# GLOBALS
#
CENSUS_YEAR = 2020
'''int : Data vintage for US Census Bureau's census shapefile (2020--2023).'''
DATA_DIR = "inputs"
'''str: Local directory for storing data files.'''
IOPub_limit = True
'''bool : Whether to limit the progress bar message to a threshold.'''
IOPub_thresh = 0.001
'''bool : The fraction of progress bar messages to show (larger == less).'''
PCS = ('esri', 102009)
'''tuple: Geopandas CRS info for North America Lambert Conformal Conic.'''
STATE_FILTER = ['PR', 'MP', 'AS', 'GU', 'VI']
'''list : A list of state and territory codes to remove from census tract.'''
ROPTS = ['BA', 'NB', 'CB', 'ST','NS','CO']
'''list : Abbreviations of available starting region options.'''
EOPTS = ['CT','CO']
'''list : Abbreviations of available ending region options.'''
WOPTS = ['A', 'Eq']
'''list : Abbreviations of available weighting options.'''
_loggers = {}
'''dict : A dictionary for storing logging instances.'''


#
# FUNCTIONS
#
def area_weighting(ee, se, name_column, ee_name_column, starting_extent):
    """Calculates conversion factors from a starting spatial extent to the
    census tract level using an area weighting method (i.e., coefficients
    proportional to shared area).

    Parameters
    ----------
    ee : geopandas.geodataframe.GeoDataFrame
        A geopandas dataframe with rows for US census tracts or counties, with column
        ee_name_column containing the GEOID (or state and county FIPS ID)
        as a unique identifier for each census tract (or county), projected into
        a coordinate reference system with linear unit 'meter'.
    se : geopandas.geodataframe.GeoDataFrame
        A geopandas dataframe with rows for starting spatial extent features,
        with column 'name_column' containing a unique identifier for each
        starting extent feature. The geodataframe should be in the same
        reference coordinate system as the census tracts.
    name_column : str
        The name of the column in se with unique identifiers for spatial
        extent  (e.g., U.S. state abbreviation).
    ee_name_column : str
        The name of the column in ee with unique identifiers for spatial
        extent  (e.g., county GEOID).
    starting_extent : str
        Spatial extent your input data is associated with.
        Currently supports one of the following,

        - "BA" (balancing authority),
        - "CB" (Coal basin),
        - "NB" (natural gas basin),
        - "ST" (state),
        - "CO" (county), and
        - "NS" (NERC sub-region).

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame containing the conversion factors from the
        spatial extents in starting extent to the census tracts or counties using the
        area weighting method.
    """
    # Re-calculate their areas (square kilometers)
    logging.info("Calculating polygon areas")
    se['SE_KM2'] = se['geometry'].area / 10**6
    ee['CEN_KM2'] = ee['geometry'].area / 10**6

    if ((starting_extent == "ST") | (starting_extent == "CO")):
        # Merge only the name column and feature areas back to census tract.
        eb = pd.merge(
            ee,
            se[[name_column, 'SE_KM2']],
            how='left',
            on=name_column
        )
        eb['AREA_KM2'] = eb['CEN_KM2']
    else:
        # Perform spatial overlay (intersection)
        # (note the warnings don't change with keep_geom_type set to false)
        logging.info("Overlaying")
        eb = gpd.overlay(se, ee, how='intersection')

        # Calculate the area of each intersection
        logging.info("Computing intersection areas")
        eb['AREA_KM2'] = eb['geometry'].area / 10**6

    df = eb.groupby(
        by=[name_column, ee_name_column]
    )[['AREA_KM2', 'SE_KM2']].agg(
        {'AREA_KM2': 'sum', 'SE_KM2': 'max'})
    df['VALUE'] = df['AREA_KM2'] / df['SE_KM2']
    return (df)


def calculate_m(starting_extent, ending_extent, weighting, state):
    """Calculate the mapping matrix, M, for translating a spatial extent
    (e.g., balancing authority, coal basin, natural gas basin, or state)
    to U.S census tracts (or counties) using either areal or equal weighting.

    Parameters
    ----------
    starting_extent : str
        Spatial extent your input data is associated with.
        Currently supports one of the following,

        - "BA" (balancing authority),
        - "CB" (Coal basin),
        - "NB" (natural gas basin),
        - "ST" (state),
        - "CO" (county), and
        - "NS" (NERC sub-region).

    ending_extent : str
        Spatial extent your input data is associated with.
        Currently supports one of the following,

        - "CT" (Census Tract),
        - "CO" (County),

    weighting : str
        Weighting method to use. Choose from,

        -   "A" (areal weighting, or impact proportional to area),
        -   "Eq" (Equal weighting, impact equal for all census tracts
            within a given spatial extent)

    state : str (optional)
        The region code for U.S. census tracts.
        Choose by state (e.g., '54' is West Virginia).
        Note that these are strings and any number less than 10 needs
        zero padding. Use 'US' for all U.S. census tracts.
        Defaults to 'US'. For state-level codes, see
        https://www2.census.gov/geo/docs/reference/codes2020/national_state2020.txt

    Returns
    -------
    tuple
        A tuple of length three.

        -   numpy.ndarray, A matrix, NxM, of conversion factors.
        -   list, A list of M census tract codes
        -   list, A list of N identifiers for the chosen spatial extent
            (balancing authority codes, eLCI coal basin codes, eLCI natural
            gas basin codes, or state code)

    Raises
    ------
    IndexError
        In the event that the data frame has duplicate rows when queried
        against the unique census ID and starting extent name ID.

    Notes
    -----
    The spatial extent mapped to is hardcoded to 2020 U.S. census tracts.
    """
    # Use the same shapefile for county and tract.
    census = get_census_geo(CENSUS_YEAR, state, make_prj=True)

    if ending_extent == "CT":
        ee = census
        ee_name_column = "GEOID"
    elif ending_extent == "CO":
        ee = tract_to_county(census)
        ee_name_column = "STCO"

    if starting_extent == "BA":
        se = get_ba_geo(correct_names=True, make_prj=True)
        name_column = 'BA_CODE'
    elif starting_extent == "CB":
        se = get_cb_geo(correct_names=True, make_prj=True)
        name_column = 'basin'
    elif starting_extent == "ST":
        se = tract_to_state(census)
        name_column = "STUSPS"
    elif starting_extent == "CO":
        se = tract_to_county(census)
        name_column = "STCO"
    elif starting_extent == "NB":
        se = get_nb_geo(correct_names=True, make_prj=True)
        name_column = "BASIN_CODE"
    elif starting_extent == "NS":
        se = get_nercsub_geo(correct_names=True, make_prj=True)
        name_column = "SUBNAME_CORRECTED"

    # Save the lengths and name lists (for later use)
    num_se = len(se)
    num_cen = len(ee)  # 84,122
    cen_list = [x for x in ee[ee_name_column]]

    # Changes based on starting extent selection
    se_list = [x for x in se[name_column]]

    if weighting == 'A':
        df = area_weighting(ee, se, name_column, ee_name_column, starting_extent)

    elif weighting == 'Eq':
        df = equal_weighting(ee, se, name_column, ee_name_column, starting_extent)

    # Initialize M matrix
    # In the new version se to census tracts, so se rows x cen cols
    logging.info("Calculating conversion matrix")
    M = np.zeros(shape=(num_se, num_cen))

    # Get totals and IO-message governor
    total = num_se * num_cen
    governor = get_io_value(total)

    # Place values into M
    for n in range(num_se):
        se_name = se_list[n]
        for m in range(num_cen):
            cen_name = cen_list[m]
            val = 0
            try:
                val = df.loc[(se_name, cen_name), 'VALUE']
            except KeyError:
                pass
            else:
                M[n,m] = val

            # Progress bar; probably need 4 decimal places
            cur_step = n*num_cen + m + 1
            if IOPub_limit and cur_step % governor == 0:
                print_progress(cur_step, total, "", 'Complete', 4)
            elif not IOPub_limit or cur_step == total:
                print_progress(cur_step, total, "", 'Complete', 4)

    # Warn about potential vacancies
    # This has been corrected for the census project
    logging.info("Checking spatial representation of conversion matrix")
    for n in range(num_se):
        se_name = se_list[n]
        if M[n,:].sum() == 0:
            logging.warning("Zero representation for se, '%s'" % se_name)
    for m in range(num_cen):
        cen_name = cen_list[m]
        if M[:,m].sum() == 0:
            logging.warning("Zero representation for CEN, '%s'" % cen_name)

    # Correct GEOID back to string for writing to file.
    cen_list = [x for x in ee[ee_name_column]]
    return (M, se_list, cen_list)


def correct_ba_geo_names(ba_geo_df):
    """Create new named column, 'BA_NAME', with mapped balancing authority
    names from HILD geospatial dataset to 2020 EIA Form 860 names.

    Notes
    -----
    UPDATE (2023-12-14): The title-case names were updated to match those
    found in the 2016--2023 names lists of the BA class.

    Source: 'elci_to_rem' Python package (NETL, 2023).

    Not all 2021 balancing authority names match the 2020 EIA Form 860 names
    and not all EIA Form 860 balancing authorities are represented in the HILD
    geo dataset (e.g., Hawaiian and Canadian authorities).

    The corrections are not as simple as making the names title case (e.g.,
    LLC, JEA, and AVBA); also, some uncertainty remains with current matches,
    such as 'Salt River Project' and 'NorthWestern Corporation'.

    Mix names that are unmatched in the geo data frame, include the following.

    -  'B.C. Hydro & Power Authority'
    -  'Hydro-Quebec TransEnergie'
    -  'Manitoba Hydro'
    -  'Ontario IESO'

    An alternative would be to match all BA areas to their representative
    BA codes, which there exists code for doing just that in scenario modeler's
    BA class.

    Parameters
    ----------
    ba_geo_df : geopandas.GeoDataFrame
        The geospatial data frame created in :func:`get_ba_geo`.

    Returns
    -------
    geopandas.GeoDataFrame
        The same as the input data frame with a new mapped column, 'BA_NAME'.
    """
    m_dict = {
        'NEW BRUNSWICK SYSTEM OPERATOR': (
            'New Brunswick System Operator'),
        'POWERSOUTH ENERGY COOPERATIVE': (
            'PowerSouth Energy Cooperative'),
        'ALCOA POWER GENERATING, INC. - YADKIN DIVISION': (
            'Alcoa Power Generating, Inc. - Yadkin Division'),
        'ARIZONA PUBLIC SERVICE COMPANY': (
            'Arizona Public Service Company'),
        'ASSOCIATED ELECTRIC COOPERATIVE, INC.': (
            'Associated Electric Cooperative, Inc.'),
        'BONNEVILLE POWER ADMINISTRATION': (
            'Bonneville Power Administration'),
        'CALIFORNIA INDEPENDENT SYSTEM OPERATOR': (
            'California Independent System Operator'),
        'DUKE ENERGY PROGRESS EAST': (
            'Duke Energy Progress East'),
        'PUBLIC UTILITY DISTRICT NO. 1 OF CHELAN COUNTY': (
            'Public Utility District No. 1 of Chelan County'),
        'CHUGACH ELECTRIC ASSN INC': (
            'Chugach Electric Assn Inc'),
        'PUD NO. 1 OF DOUGLAS COUNTY': (
            'PUD No. 1 of Douglas County'),
        'DUKE ENERGY CAROLINAS': (
            'Duke Energy Carolinas'),
        'EL PASO ELECTRIC COMPANY': (
            'El Paso Electric Company'),
        'ELECTRIC RELIABILITY COUNCIL OF TEXAS, INC.': (
            'Electric Reliability Council of Texas, Inc.'),
        'ELECTRIC ENERGY, INC.': (
            'Electric Energy, Inc.'),
        'FLORIDA POWER & LIGHT COMPANY': (
            'Florida Power & Light Co.'),
        'DUKE ENERGY FLORIDA INC': (
            'Duke Energy Florida, Inc.'),
        'GAINESVILLE REGIONAL UTILITIES': (
            'Gainesville Regional Utilities'),
        'CITY OF HOMESTEAD': (
            'City of Homestead'),
        'IDAHO POWER COMPANY': (
            'Idaho Power Company'),
        'IMPERIAL IRRIGATION DISTRICT': (
            'Imperial Irrigation District'),
        'JEA': (
            'JEA'),
        'LOS ANGELES DEPARTMENT OF WATER AND POWER': (
            'Los Angeles Department of Water and Power'),
        'LOUISVILLE GAS AND ELECTRIC COMPANY AND KENTUCKY UTILITIES': (
            'Louisville Gas and Electric Company and Kentucky Utilities Company'),
        'NORTHWESTERN ENERGY (NWMT)': (
            'NorthWestern Corporation'),
        'NEVADA POWER COMPANY': (
            'Nevada Power Company'),
        'ISO NEW ENGLAND INC.': (
            'ISO New England'),
        'NEW SMYRNA BEACH, UTILITIES COMMISSION OF': (
            'Utilities Commission of New Smyrna Beach'),
        'NEW YORK INDEPENDENT SYSTEM OPERATOR': (
            'New York Independent System Operator'),
        'OHIO VALLEY ELECTRIC CORPORATION': (
            'Ohio Valley Electric Corporation'),
        'PACIFICORP - WEST': (
            'PacifiCorp West'),
        'PACIFICORP - EAST': (
            'PacifiCorp East'),
        'GILA RIVER POWER, LLC': (
            'Gila River Power, LLC'),
        'FLORIDA MUNICIPAL POWER POOL': (
            'Florida Municipal Power Pool'),
        'PUBLIC UTILITY DISTRICT NO. 2 OF GRANT COUNTY, WASHINGTON': (
            'Public Utility District No. 2 of Grant County, Washington'),
        'PJM INTERCONNECTION, LLC': (
            'PJM Interconnection, LLC'),
        'PORTLAND GENERAL ELECTRIC COMPANY': (
            'Portland General Electric Company'),
        'AVANGRID RENEWABLES LLC': (
            'Avangrid Renewables, LLC'),
        'PUBLIC SERVICE COMPANY OF COLORADO': (
            'Public Service Company of Colorado'),
        'PUBLIC SERVICE COMPANY OF NEW MEXICO': (
            'Public Service Company of New Mexico'),
        'PUGET SOUND ENERGY': (
            'Puget Sound Energy, Inc.'),
        'BALANCING AUTHORITY OF NORTHERN CALIFORNIA': (
            'Balancing Authority of Northern California'),
        'SALT RIVER PROJECT': (
            'Salt River Project Agricultural Improvement and Power District'),
        'SEATTLE CITY LIGHT': (
            'Seattle City Light'),
        'SOUTH CAROLINA ELECTRIC & GAS COMPANY': (
            'Dominion Energy South Carolina, Inc.'),
        'SOUTH CAROLINA PUBLIC SERVICE AUTHORITY': (
            'South Carolina Public Service Authority'),
        'SOUTHWESTERN POWER ADMINISTRATION': (
            'Southwestern Power Administration'),
        'SOUTHERN COMPANY SERVICES, INC. - TRANS': (
            'Southern Company Services, Inc. - Trans'),
        'CITY OF TACOMA, DEPARTMENT OF PUBLIC UTILITIES, LIGHT DIVISION': (
            'City of Tacoma, Department of Public Utilities, Light Division'),
        'CITY OF TALLAHASSEE': (
            'City of Tallahassee'),
        'TAMPA ELECTRIC COMPANY': (
            'Tampa Electric Company'),
        'TENNESSEE VALLEY AUTHORITY': (
            'Tennessee Valley Authority'),
        'TURLOCK IRRIGATION DISTRICT': (
            'Turlock Irrigation District'),
        'HAWAIIAN ELECTRIC CO INC': ( #I can't find this one!
            'Hawaiian Electric Co Inc'),
        'WESTERN AREA POWER ADMINISTRATION UGP WEST': (
            'Western Area Power Administration - Upper Great Plains West'),
        'AVISTA CORPORATION': (
            'Avista Corporation'),
        'SEMINOLE ELECTRIC COOPERATIVE': (
            'Seminole Electric Cooperative'),
        'TUCSON ELECTRIC POWER COMPANY': (
            'Tucson Electric Power'),
        'WESTERN AREA POWER ADMINISTRATION - DESERT SOUTHWEST REGION': (
            'Western Area Power Administration - Desert Southwest Region'),
        'WESTERN AREA POWER ADMINISTRATION - ROCKY MOUNTAIN REGION': (
            'Western Area Power Administration - Rocky Mountain Region'),
        'SOUTHEASTERN POWER ADMINISTRATION': (
            'Southeastern Power Administration'),
        'NEW HARQUAHALA GENERATING COMPANY, LLC - HGBA': (
            'New Harquahala Generating Company, LLC'),
        'GRIFFITH ENERGY, LLC': (
            'Griffith Energy, LLC'),
        'NATURENER POWER WATCH, LLC (GWA)': (
            'NaturEner Power Watch, LLC'),
        # misnamed the following one by mistake and I can't find it.
        # Corrected
        'GRIDFORCE SOUTH': (
            'Gridforce South'),
        'MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR, INC..': (
            'Midcontinent Independent System Operator, Inc.'),
        'ARLINGTON VALLEY, LLC - AVBA': (
            'Arlington Valley, LLC'),
        'DUKE ENERGY PROGRESS WEST': (
            'Duke Energy Progress West'),
        'GRIDFORCE ENERGY MANAGEMENT, LLC': (
            'Gridforce Energy Management, LLC'),
        'NATURENER WIND WATCH, LLC': (
            'NaturEner Wind Watch, LLC'),
        'SOUTHWEST POWER POOL': (
            'Southwest Power Pool'),
    }
    # Also, the excel sheet has 78 entries, not 71.
    # Number wise, compare to data in HIFLD website, not the excel sheet.
    # https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::control-areas/about
    logging.info("Correcting balancing authority names")
    ba_geo_df['BA_NAME'] = ba_geo_df['NAME'].map(m_dict)

    return ba_geo_df


def correct_cb_geo_names(basin_shp):
    """Correct the coal basin names in the geodatabase.

    Create new named column, 'basin', with mapped coal basin names to match
    eLCI names. Also drop any basins with names that don't match any of these.

    Notes
    -----
    Source: 'elci_to_rem' Python package (NETL, 2023).

    Parameters
    ----------
    basin_shp : geopandas.GeoDataFrame
        The geospatial data frame created in :func:`get_cb_geo`.

    Returns
    -------
    geopandas.GeoDataFrame
        The same as the input data frame with a new mapped column, 'basin'.
    """
    bas = []
    for b in basin_shp.BASIN_AREA:
        if b == 'Central Appalachian': bas.append('CA')
        elif b == 'Arkoma' \
            or b == 'Cherokee' \
            or b == 'Forest City' \
            or b == 'SW Coal Region':
                bas.append('CI')
        elif b == 'Gulf Coast': bas.append('GL')
        elif b == 'Illinois': bas.append('IB')
        elif b == 'Williston Basin': bas.append('L')
        elif b == 'Northern Appalachian': bas.append('NA')
        elif b == 'Powder River': bas.append('PRB')
        elif b == 'Big Horn Basin' \
            or b == 'Denver Basin' \
            or b == 'Goshen Hole Field' \
            or b == 'Greater Green River' \
            or b == 'Hannah-Carbon Basin' \
            or b == 'HenryMtns-SWColorado' \
            or b == 'Jackson Hole Field' \
            or b == 'KaiparowitsBlackMesa' \
            or b == 'North Central' \
            or b == 'Raton' \
            or b == 'Rock Creek Field' \
            or b == 'San Juan' \
            or b == 'Uinta-Piceance-Park' \
            or b == 'Wind River' \
            or b == 'Wyoming overthrust':
                bas.append('RM')
        elif b == 'Black Warrior': bas.append('SA')
        elif b == 'Pacific': bas.append('WNW')
        else: bas.append('no_emf')

    basin_shp['basin'] = bas
    basin_shp = basin_shp[basin_shp['basin'] != 'no_emf']
    basin_shp = basin_shp.dissolve(by='basin')
    basin_shp['basin'] = basin_shp.index

    return basin_shp


def correct_nb_geo_names(nb_geo_df):
    """Create new named column, 'basin', with mapped natural gas basin
    names to match eLCI basin names.

    Notes
    -----
    Source: 'elci_to_rem' Python package (NETL, 2023).

    Parameters
    ----------
    nb_geo_df : geopandas.GeoDataFrame
        The geospatial data frame created in :func:`get_nb_geo`.

    Returns
    -------
    geopandas.GeoDataFrame
        The same as the input data frame with a new mapped column, 'basin'.
    """
    # Rename NG basins to match eLCI basin names (see unique stage code names
    # in eLCI output for source).
    # NOTE: it's where multiple basin names are reclassified that duplicates
    # occur later.
    bas = []
    for b in nb_geo_df.BASIN_NAME:
        if b == 'Anadarko Basin':
            bas.append('Anadarko')
        elif b == 'Appalachian Basin' or b == 'Appalachian Basin (Eastern Overthrust Area)':
            bas.append('Appalachian')
        elif b == 'Arkla Basin':
            bas.append('Arkla')
        elif b == 'Arkoma Basin':
            bas.append('Arkoma')
        elif b == 'East Texas Basin':
            bas.append('East Texas')
        elif b == 'Forth Worth Basin' \
            or b == 'Fort Worth Syncline' \
            or b == 'Bend Arch':
                bas.append('Fort Worth Basin')
        elif b == 'Green River Basin':
            bas.append('Green River')
        elif b == 'Gulf Coast Basin (LA, TX)':
            bas.append('Gulf')
        elif b == 'Permian Basin':
            bas.append('Permian')
        elif b == 'Piceance Basin':
            bas.append('Piceance')
        elif b == 'San Juan Basin':
            bas.append('San Juan')
        elif b == 'South Oklahoma Folded Belt':
            bas.append('South Oklahoma')
        elif b == 'Strawn Basin':
            bas.append('Strawn')
        elif b == 'Uinta Basin':
            bas.append('Uinta')
        else:
            bas.append('no_emf')
    logging.info("Correcting coal basin names")
    nb_geo_df['basin'] = bas
    nb_geo_df = nb_geo_df[nb_geo_df['basin'] != 'no_emf']
    nb_geo_df = nb_geo_df.dissolve(by='basin')
    nb_geo_df['basin'] = nb_geo_df.index

    return nb_geo_df

def correct_ns_geo_names(ns_geo_df):
    """Correct spaces in NERC subregion names to conform to the M matrix
    space-separated naming scheme.

    Notes
    -----
    Creates a new named column, 'SUBNAME_CORRECTED'.

    The four names at fault and their replacements are:

    - 'CA-MX US' ('CA-MX')
    - 'NEW ENGLAND' ('NE')
    - 'NEW YORK' ('NY')
    - 'MRO US' ('MRO')

    Parameters
    ----------
    ns_geo_df : geopandas.GeoDataFrame
        The geospatial data frame created in :func:`get_nercsub_geo`.

    Returns
    -------
    geopandas.GeoDataFrame
        The same as the input data frame with a new mapped column,
        'SUBNAME_CORRECTED'.
    """
    m_dict = {
        'CA-MX US': (
            'CA-MX'),
        'NEW ENGLAND': (
            'NE'),
        'NEW YORK': (
            'NY'),
        'MRO US': (
            'MRO')
    }
    # Also, the excel sheet has 78 entries, not 71. Do I add the others?
    # Fixed! Number wise, compare to the data in the HIFLD website, not the
    # excel sheet.
    # https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::control-areas/about
    logging.info("Correcting NERC subregion names")
    ns_geo_df['SUBNAME_CORRECTED'] = ns_geo_df['SUBNAME'].replace(m_dict)

    return ns_geo_df


def download_file(url, filepath):
    """Download a file from the web.

    Parameters
    ----------
    url : str
        A web address that points to a file.
    filepath : str
        A file path (including the file name) to where the local copy of the
        file should be downloaded.

    Returns
    -------
    bool
        Whether the requests download was successful.
    """
    r = requests.get(url)
    if r.ok:
        with open(filepath, 'wb') as f:
            f.write(r.content)

    return r.ok


def equal_weighting(ee, se, name_column, ee_name_column, starting_extent):
    """Calculates conversion factors from larger spatial extents to the census
    tract level using an equal weighting method (assuming equal impact for
    all census tracts in a given spatial extent).

    Parameters
    ----------
    ee : geopandas.geodataframe.GeoDataFrame
        A geopandas dataframe with rows for US census tracts or counties, with column
        ee_name_column containing the GEOID (or state and county FIPS ID)
        as a unique identifier for each census tract (or county), projected into
        a coordinate reference system with linear
        unit 'meter'.
    se : geopandas.geodataframe.GeoDataFrame
        A geopandas dataframe with rows for starting spatial extent features,
        with column 'name_column' containing a unique identifier for each
        starting extent feature. The geodataframe should be in the same
        reference coordinate system as the census tracts.
    name_column : str
        The name of the column in se with unique identifiers for spatial
        extent  (e.g., U.S. state abbreviation).
    starting_extent : str
        Spatial extent your input data is associated with.
        Currently supports one of the following,

        - "BA" (balancing authority),
        - "CB" (Coal basin),
        - "NB" (natural gas basin),
        - "ST" (state),
        - "CO" (county), and
        - "NS" (NERC sub-region).

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame containing the conversion factors from the
        spatial extents in starting extent to the census tracts using the
        equal weighting method.
    """
    if ((starting_extent == "ST") | (starting_extent == "CO")):
        # The census data frame already has both starting and ending
        # features (e.g., STUSPS and FIPS); no overlay required.
        eb = ee
    else:
        logging.info("Intersecting data frames...")
        eb = gpd.overlay(se, ee, how='intersection')

    logging.info("Counting intersections")
    eb['count'] = eb.groupby(name_column)[name_column].transform('count')

    # TODO: Check whether [['count']] is needed here between the groupby
    # and the agg.
    df = eb.groupby(
        by=[name_column, ee_name_column]
    )[['count']].agg({'count': 'sum'})

    logging.info("Normalizing counts")
    df['VALUE'] = 1/df['count'].values

    return df


def get_ba_geo(correct_names=False, make_prj=False):
    """Create a geospatial data frame for U.S. control areas (i.e., balancing
    authorities).

    Run this method once to download a local copy of the GeoJSON.
    Subsequent runs of this method attempt to read the local file rather
    than re-download the file. The file name is "control_areas.geojson" and
    is saved in the DATA_DIR directory (e.g., ./data).

    Notes
    -----
    Source: 'elci_to_rem' Python package (NETL, 2023).

    The API referenced in this method links to 2021 control areas, which were
    updated in 2022.

    When correcting BA names, there are a few that do not match the EIA 923
    names, which include Chugach Electric Assn Inc, Avangrid Renewables LLC,
    and Hawaiian Electric Co Inc.

    Source, "Control Areas" from Homeland Infrastructure Foundation Level
    Database (HIFLD). Online [1]_.

    [1] https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::control-areas/about

    Parameters
    ----------
    correct_names : bool, optional
        Whether to create a new named column, 'BA_NAME', with balancing
        authority names mapped to the EIA Form 860 balancing authority area
        names, defaults to false.
    make_prj : bool, optional
        Whether to project the coordinate reference system to PCS.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame of polygon areas representing the U.S.
        electricity control areas (i.e., balancing authorities).

        Columns include:

        - 'OBJECTID',
        - 'ID',
        - 'NAME',
        - 'ADDRESS',
        - 'CITY',
        - 'STATE',
        - 'ZIP',
        - 'TELEPHONE',
        - 'COUNTRY',
        - 'NAICS_CODE',
        - 'NAICS_DESC',
        - 'SOURCE',
        - 'SOURCEDATE',
        - 'VAL_METHOD',
        - 'VAL_DATE',
        - 'WEBSITE',
        - 'YEAR',
        - 'PEAK_MONTH',
        - 'AVAIL_CAP',
        - 'PLAN_OUT',
        - 'UNPLAN_OUT',
        - 'OTHER_OUT',
        - 'TOTAL_CAP',
        - 'Value_of_interest',
        - 'MIN_LOAD',
        - 'SHAPE__Area',
        - 'SHAPE__Length',
        - 'GlobalID',
        - 'geometry'
    """
    # NOTE: consider including comma-separated list of outFields, as not all
    # are needed and/or used.
    ba_api_url = (
        "https://services1.arcgis.com/"
        "Hp6G80Pky0om7QvQ/arcgis/rest/services/Control_Areas_gdb/"
        "FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson")
    ba_file = "control_areas.geojson"
    ba_path = os.path.join(DATA_DIR, ba_file)

    # Check to make sure data directory exists before attempting download
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Use existing file if available:
    if not os.path.isfile(ba_path):
        logging.info("Downloading balancing authority GEOJSON")
        download_file(ba_api_url, ba_path)

    # Read GeoJSON and correct BA area names (if requested)
    logging.info("Reading balancing authority GEOJSON")
    gdf = gpd.read_file(ba_path)
    if correct_names:
        gdf = correct_ba_geo_names(gdf)
        gdf = map_ba_codes(gdf)
    if make_prj:
        logging.info("Projecting balancing authority map")
        gdf = gdf.to_crs(PCS)

    return gdf


def get_ba_map():
    """Return a dictionary of balancing authority names and their abbreviations

    Notes
    -----
    Source: 'elci_to_rem' Python package (NETL, 2023). Updated December 2023
    to utilize BA class (rather than electricitylci.combinatory.ba_codes).

    Returns
    -------
    dict
        A dictionary with keys of balancing authority names (as per EIA 923)
        and values of abbreviations from 2016--2022.
    """
    ba_map = {}
    df = read_ba_codes()
    for ba_code, row in df.iterrows():
        ba_name = row['BA_Name']
        ba_map[ba_name] = ba_code

    return ba_map


def get_cb_geo(correct_names=False, make_prj=False):
    """Create a geospatial data frame for U.S. coal basins.

    Notes
    -----
    Source: EIA
    [1] https://statics.teams.cdn.office.net/evergreen-assets/safelinks/1/atp-safelinks.html

    Parameters
    ----------
    correct_names : bool, optional
        Whether to create a new named column, 'basin', with coal basin names
        mapped to the eLCI basin names, and drop any rows which cannot be
        mapped to one of these defaults to false.
    make_prj : bool, optional
        Whether to project the coordinate reference system to PCS.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame of polygon areas representing the U.S.
        coal basins.

        Columns include:

        - 'RSC_04_TCF',
        - 'BASIN_AREA',
        - 'RSC_06_TCF',
        - 'geometry'
        - 'basin' (if correct_names=True)
    """
    # NOTE: consider including comma-separated list of outFields, as not all
    # are needed and/or used.
    ba_api_url = (
        "https://www.eia.gov/maps/map_data/cbm_4shps.zip")
    ba_file = "cbm_4shps.zip"
    ba_path = os.path.join(DATA_DIR, ba_file)

    # Check to make sure data directory exists before attempting download
    if not os.path.isdir(DATA_DIR):
        os.mkdirs(DATA_DIR)

    # Use existing file if available:
    if not os.path.isfile(ba_path):
        download_file(ba_api_url, ba_path)

    # Read GeoJSON and correct BA area names (if requested)
    gdf = gpd.read_file(ba_path, layer = 'CBMbasins_resources_2006')

    if make_prj:
        logging.info("Projecting census tract map")
        gdf = gdf.to_crs(PCS)

    if correct_names:
        gdf = correct_cb_geo_names(gdf)

    return gdf


def get_census_geo(year, region, make_prj=False):
    """Read US census track region shapefile for a given year and region.

    Notes
    -----
    The GIS shapefiles are published by the U.S. Census Bureau.
    Census tracts are areas of land within the United States that
    are home to roughly 4,000 people each.
    They always follow county lines, and may also follow boundaries
    such as municipal lines, rivers, and roads. Due to the small
    size of census tracts, this is a large shapefile, containing
    85,187 rows in the 2020 census.

    Source: https://www2.census.gov/geo/tiger/GENZ2020/shp/

    Parameters
    ----------
    year : int
        The year to pull the geospatial data set.
        Valid years include 2020--2023.
    region : str
        The region code.
        If state, it's the state number (e.g., '54' is West Virginia).
        Note that state numbers less than 10 should be zero padded (e.g. '08').
        If U.S., use 'US'
    make_prj : bool, optional
        Whether to project the geometry to 2D using the PCS coordinate
        system (e.g., North America Lambert Conformal Conic).

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame of polygon areas representing the U.S.
        census tracts, with columns,

        - STATEFP (str), two digit state ID (zero padded)
        - COUNTYFP (str), three digit county ID (zero padded)
        - TRACTCE (str), six digit tract ID (zero padded)
        - AFFGEOID (str), 20-digit ID
        - GEOID (str), 11-digit ID of STATEFP, COUNTYFP, and TRACTCE
        - NAME (str), tract short name
        - NAMELSAD (str), tract long name
        - STUSPS (str), two-character state abbreviation
        - STATE_NAME (str), state name
        - LSAD (str), legal statistical area description code ('CT')
        - ALAND (int), land area
        - AWATER (int), water area

    Raises
    ------
    OSError
        The zipped shapefile is not publicly accessible. This error raises
        when the local file is not found.
    """
    # Create the census file name, (e.g., "cb_2020_us_tract_500k.zip")
    if isinstance(region, int):
        region = "%02d" % region
    region = region.lower()
    cen_file = "cb_%d_%s_tract_500k.zip"  % (year, region)
    cen_path = os.path.join(DATA_DIR, cen_file)

    # Handle missing census data files.
    if not os.path.isfile(cen_path):
        logging.info("Downloading the census file, %s" % cen_file)
        cen_url = "https://www2.census.gov/geo/tiger/GENZ%d/shp/%s" % (
            year, cen_file)
        _worked = download_file(cen_url, cen_path)
        if _worked:
            logging.info("Census tract file downloaded!")

    logging.info("Reading census tract shapefile")
    gdf = gpd.read_file(cen_path)

    # Remove non-states from data frame
    gdf = gdf.query("STUSPS not in @STATE_FILTER")

    if make_prj:
        logging.info("Projecting census tract map")
        gdf = gdf.to_crs(PCS)

    return gdf


def get_io_value(max_val):
    """The value used as a modulus to limit the number of messages printed by
    the progress bar.

    Parameters
    ----------
    max_val : int
        The total number of messages.

    Returns
    -------
    int
        Nearest hundredth of a given fraction of the total messages.
        Depends on global parameter, `IOPub_thresh`.

    Examples
    --------
    >>> get_io_value(123456789)
    123500
    """
    return int(round(max_val*IOPub_thresh, -2))


def get_logger(name='root'):
    """Convenience function for retrieving loggers by name."""
    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        rec_format = (
            "%(asctime)s, %(name)s.%(funcName)s: "
            "%(message)s")
        formatter = logging.Formatter(rec_format, datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.handlers[0].stream = sys.stdout
        _loggers[name] = logger

    return _loggers[name]


def get_m(starting_extent, ending_extent, weighting, state='US'):
    """Return the mapping matrix and name lists for starting extent to census
    (or county) translation.

    Parameters
    ----------
    starting_extent : str
        The region abbreviation for the starting extent.
        Available options include,

        - 'BA' balancing authority
        - 'CB' coal basins
        - 'NG' natural gas basins
        - 'ST' U.S. states

    ending_extent : str
        Spatial extent your input data is associated with.
        Currently supports one of the following,

    weighting : str
        The weighting method.
        Available options include,

        - 'A' areal weighting
        - 'Eq' equal weighting
    state : str (optional)
        The region code for U.S. census tracts.
        Choose by state (e.g., '54' is West Virginia)---
        note that these are strings and any number less than 10 needs
        a zero padding.
        Use 'US' for all U.S. census tracts.
        Defaults to 'US'.

    Returns
    -------
    tuple
        A tuple of length three.

        -   numpy.ndarray, A matrix, NxM, of conversion factors.
        -   list, A list of M census tract (or county) codes
        -   list, A list of N identifiers for the chosen spatial extent
            (balancing authority codes, eLCI coal basin codes, eLCI natural
            gas basin codes, or state code).

    Notes
    -----
    This method reads m from file (if it exists); otherwise, it calculates m
    and saves it to file before returning it.

    The naming scheme for the text file is:

    - "m\\_"
    - starting extent code (e.g., 'BA')
    - weighting code (e.g., 'A')
    - ".txt"

    For example: m_BA_A.txt (area-weighted from balancing authority matrix).
    """
    # Error handling
    if starting_extent.lower() not in [x.lower() for x in ROPTS]:
        raise ValueError(
            "Starting region option, '%s', is not available." % starting_extent)
    if ending_extent.lower() not in [x.lower() for x in EOPTS]:
        raise ValueError(
            "Ending region option, '%s', is not available." % ending_extent)
    if weighting.lower() not in [x.lower() for x in WOPTS]:
        raise ValueError(
            "Weighting option, '%s', is not available." % weighting)
    logging.info(
        "Getting M for %s region using %s weighting method." % (
            starting_extent, weighting
        )
    )

    # The m.txt file stores the conversion matrix.
    # It's location is defined here, but if it doesn't exist,
    # then the code will create the matrix and save it to the file.
    map_file_us_ct = os.path.join(
        DATA_DIR, "m_" + starting_extent + "_" + weighting + ".txt"
    )

    map_file_ct = os.path.join(
        DATA_DIR, "m_" + starting_extent + "_" + state +"_" + weighting + ".txt"
    )

    map_file = os.path.join(
        DATA_DIR, "m_" + starting_extent + "_" + ending_extent + "_" + state +"_" + weighting + ".txt"
    )



    # If the matrix file exists, read it; otherwise, create it.
    if os.path.isfile(map_file):
        logging.info("Reading mapping file")
        M, ba_list, cen_list = read_m(map_file)
    elif (os.path.isfile(map_file_us_ct) & (state == 'US') & (ending_extent == 'CT')):
        logging.info("Reading mapping file")
        M, ba_list, cen_list = read_m(map_file_us_ct)
    elif (os.path.isfile(map_file_ct) & (ending_extent == 'CT')):
        logging.info("Reading mapping file")
        M, ba_list, cen_list = read_m(map_file_ct)
    else:
        logging.info("Creating mapping matrix")
        M, ba_list, cen_list = calculate_m(starting_extent, ending_extent, weighting, state)
        save_m(M, ba_list, cen_list, map_file)
        logging.info("Saved map to '%s'" % map_file)

    return (M, ba_list, cen_list)


def get_nb_geo(correct_names=False, filter_basins=True, make_prj=False):
    """Create a geospatial data frame for U.S. natural gas basins.

    Notes
    -----
    Source: EIA
    [1] https://statics.teams.cdn.office.net/evergreen-assets/safelinks/1/atp-safelinks.html

    Parameters
    ----------
    correct_names : bool, optional
        Whether to create a new named column, 'basin', with natural gas basin
        names mapped to the eLCI basin names, and drop any rows which cannot be
        mapped to one of these; defaults to false.
    filter_basins : bool, optional
        Filters basins by their code to represent natural gas basins that
        align with NETL's electricity baseline.
    make_prj : bool, optional
        Whether to project the coordinate reference system to PCS.
        Defaults to false.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame of polygon areas representing the U.S.
        natural gas basins.

        Columns include:

        - 'BASIN_CODE',
        - 'BASIN_NAME',
        - 'geometry'
        - 'basin' (if correct_names=True)
    """
    _api_url = "https://edg.epa.gov/data/Public/OAR/OAP/Basins_Shapefile.zip"
    _file = os.path.basename(_api_url)
    _path = os.path.join(DATA_DIR, _file)

    # Check to make sure data directory exists before attempting download
    if not os.path.isdir(DATA_DIR):
        os.mkdirs(DATA_DIR)

    # Use existing file if available:
    if not os.path.isfile(_path):
        logging.info("Downloading natural gas basins shapefile")
        download_file(_api_url, _path)

    # Read shapefile and filter for NB basins of interest.
    # NOTE: should work on a zip file
    gdf = gpd.read_file(_path)

    # These 31 basins were identified and matched to electricity baseline;
    # see 'Update 2021' section in 05.1-xds-ng_basin_county.ipynb found here:
    # https://github.com/KeyLogicLCA/elci_to_rfm/blob/main/notebooks/
    if filter_basins:
        basins_of_interest =  [
            '160', '160A', '210', '220', '230',
            '260', '305', '345', '350', '355',
            '360', '375', '395', '400', '415',
            '420', '425', '430', '435', '450',
            '455', '507', '515', '530', '535',
            '540', '575', '580', '585', '595',
            '730'
        ]
        gdf = gdf.loc[gdf['BASIN_CODE'].isin(basins_of_interest), :].copy()

    if make_prj:
        logging.info("Projecting census tract map")
        gdf = gdf.to_crs(PCS)
    if correct_names:
        gdf = correct_nb_geo_names(gdf)

    return gdf


def get_nercsub_geo(correct_names=False, make_prj=False):
    """IN PROGRESS

    Create a geospatial data frame for U.S. Nerc subregions

    Run this method once to download a local copy of the GeoJSON.
    Subsequent runs of this method attempt to read the local file rather
    than re-download the file. The file name is "control_areas.geojson" and
    is saved in the DATA_DIR directory (e.g., ./data).

    Notes
    -----
    The API referenced in this method links to NERC subregions, which were
    updated in 2022.

    Parameters
    ----------
    correct_names : bool, optional
        Whether to create a new named column, 'SUBNAME_CORRECTED', with the
        four subnames which have a space replaced with new names which do not,
        to enable them to be saved to and read from a space-seperated format.
    make_prj : bool, optional
        Whether to project the coordinate reference system to PCS.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame of polygon areas representing the U.S.
        NERC subregions.

        Columns include:

        - 'OBJECTID',
        - 'ID',
        - 'NAME',
        - 'ADDRESS',
        - 'CITY',
        - 'STATE',
        - 'ZIP',
        - 'COUNTRY',
        - 'SOURCE',
        - 'SOURCEDATE',
        - 'VAL_METHOD',
        - 'VAL_DATE',
        - 'WEBSITE',
        - 'SUBNAME',
        - 'SHAPE__Area',
        - 'SHAPE__Length',
        - 'geometry'
    """
    # NOTE: consider including comma-separated list of outFields, as not all
    # are needed and/or used.
    ba_api_url = (
        "https://services1.arcgis.com/"
        "Hp6G80Pky0om7QvQ/arcgis/rest/services/NERC_Regions/"
        "FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson")
    ba_file = "nerc_subregions.geojson"
    ba_path = os.path.join(DATA_DIR, ba_file)

    # Check to make sure data directory exists before attempting download
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Use existing file if available:
    if not os.path.isfile(ba_path):
        logging.info("Downloading NERC GEOJSON")
        download_file(ba_api_url, ba_path)

    # Read GeoJSON and correct BA area names (if requested)
    logging.info("Reading NERC GEOJSON")
    gdf = gpd.read_file(ba_path)

    if make_prj:
        logging.info("Projecting NERC map")
        gdf = gdf.to_crs(PCS)
    if correct_names:
        gdf = correct_ns_geo_names(gdf)
    return gdf


def map_ba_codes(df):
    """Map balancing authority abbreviation codes based on EIA Form 930 naming.

    The goal of including the BA codes is to make the matrix file easier to
    manage. The BA names are not standardized and can be difficult to map
    (e.g., capitalization, periods after Inc. and Corp., the placement of
    'City of' before or after a place).

    This method is used in conjunction with :func:`get_ba_geo` to create
    a new column, 'BA_CODE'.

    Parameters
    ----------
    df : pandas.DataFrame
        A data frame with column, 'Subregion' or 'BA_NAME' used to match
        against balancing authority abbreviation map.

    Returns
    -------
    pandas.DataFrame
        The same as the sent data frame with a new column, "BA_CODE".
    """
    logging.info("Mapping balancing authority names to codes")
    m_col = 'Subregion'
    if 'Subregion' not in df.columns and 'BA_NAME' in df.columns:
        m_col = 'BA_NAME'
    elif 'Subregion' not in df.columns and 'BA_NAME' not in df.columns:
        logging.warning("No matching column for BA codes!")

    ba_map = get_ba_map()
    df['BA_CODE'] = df[m_col].map(ba_map)
    logging.info("%d mis-matched BA codes" % df['BA_CODE'].isna().sum())

    return df


def print_progress(iteration, total, prefix='', suffix='', decimals=0,
                   bar_length=44):
    """Create a terminal progress bar.

    Parameters
    ----------
    iteration : int
        Current iteration.
    total : int
        Total iterations.
    prefix : str, optional
        Prefix string, defaults to empty string.
    suffix : str, optional
        Suffix string, defaults to empty string.
    decimals : int, optional
        The number of decimal places to display in percent complete.
        Defaults to zero (i.e., whole integer).
    bar_length : int, optional
        The character length of the progress bar, defaults to 44.

    Notes
    -----
    Reference:
        "Python Progress Bar" by Aubrey Taylor (c) 2020.
        https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    Examples
    --------
    >>> total = 200
    >>> for i in range(2):
    ...     for j in range(100):
    ...         cur_step = (2*i) + (j+1)
    ...         print_progress(cur_step, total, suffix='Complete')
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write(
        '\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def read_ba_codes():
    """Read balancing authority short codes from EIA930 reference table.

    The Excel workbook referenced may be found here:
    https://www.eia.gov/electricity/930-content/EIA930_Reference_Tables.xlsx

    Downloads the Excel workbook for offline access, and adds three missing
    BA entries (GRIS, CEA, HECO).

    Columns for sheet, "BAs" (as of 2024), include:

    - BA Code (str)
    - BA Name (str)
    - Time Zone (str): For example, "Eastern," "Central" or "Pacific"
    - Region/Country Code (str): EIA region code
    - Region/Country Name (str): EIA region name
    - Generation Only BA (str): "Yes" or "No"
    - Demand by BA Subregion (str): "Yes" or "No"
    - U.S. BA (str): "Yes" or "No"
    - Active BA (str): "Yes" or "No"
    - Activation Date: (str/NA): mostly empty, a few years are available
    - Retirement Date (str/NA): mostly empty, a few years are available

    Returns
    -------
    pandas.DataFrame
        Index is set to BA_Acronym (BA Code).
        Column renames include 'BA_Name' (BA Name), 'EIA_Region_Abbr', and
        'EIA_Region'.
    """
    eia_ref_file = "EIA930_Reference_Tables.xlsx"
    eia_ref_path = os.path.join("inputs", eia_ref_file)
    eia_ref_url = "https://www.eia.gov/electricity/930-content/EIA930_Reference_Tables.xlsx"

    if not os.path.isfile(eia_ref_path):
        download_file(eia_ref_url, eia_ref_path)

    df = pd.read_excel(eia_ref_path)
    df = df.rename(columns={
        'BA Code': 'BA_Acronym',
        'BA Name': 'BA_Name',
        'Region/Country Code': 'EIA_Region_Abbr',
        'Region/Country Name': 'EIA_Region',
    })

    # HOTFIX: add missing BA acronyms:
    tmp_dict = {
          'BA_Acronym': ['GRIS', 'CEA', 'HECO'],
          'BA_Name': ['Gridforce South',
                      'Chugach Electric Assn Inc',
                      'Hawaiian Electric Co Inc'],
    }
    df = pd.concat([df, pd.DataFrame(tmp_dict)])
    df = df.set_index("BA_Acronym")

    return df


def read_m(m_file):
    """Read a mapping matrix from file that was saved using :func:`save_m`.

    Parameters
    ----------
    m_file : str
        A file path to a plain text mapping file (e.g., m.txt).

    Returns
    -------
    tuple
        A tuple of length three.

        - numpy.ndarray : mapping matrix (nxm)
        - list : n region names (or abbreviations)
        - list : m census tract GEOIDs (or county FIPs IDs)
    """
    logging.info("Parsing the matrix file")

    # Just read the headerlines; thanks John La Rooy (2009)
    # https://stackoverflow.com/a/1767589
    with open(m_file) as f:
        head = [next(f) for _ in range(2)]

    # DO NOT SORT LISTS! Order matters.
    m_names = head[0].strip()
    m_names = m_names.split(" ")
    b_names = head[1].strip()
    b_names = b_names.split(" ")

    # https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
    m = np.loadtxt(m_file, delimiter=" ", skiprows=3)

    return (m, b_names, m_names)


def save_m(m, b_names, m_names, m_file):
    """Write the mapping matrix to file.

    Notes
    -----
    The output file format is plain text (.txt) with space-delimited contents
    saved using the following as a guide:

    -   Line 1: Names (or abbreviations) of 'from' region
    -   Line 2: Census tract GEOIDS (or county FIPs IDs)
    -   Line 3: (blank)
    -   Lines 4-n: Rows of the mapping matrix (float, no scaling)

    Parameters
    ----------
    m : numpy.ndarray
        The 2D mapping matrix.
    b_names : list
        Names (or abbreviations) of 'from' regions.
    m_names : list
        List of census tract GEOIDs (or county FIPs IDs).
    m_file : str
        The file path for the output text file.

    Raises
    ------
    IndexError
        Thrown when columns or rows do not match their respective list lengths.
    """
    num_m = len(m_names)
    num_b = len(b_names)
    num_r, num_c = m.shape

    if num_r != num_b:
        raise IndexError(
            "The number of rows should match the number of "
            "regions! (%d != %d)" % (num_r, num_b))

    if num_c != num_m:
        raise IndexError(
            "The number of columns should match the number of "
            "census tracts or counties! (%d != %d)" % (num_c, num_m))

    b_names = [b_name.replace(' ', '_') for b_name in b_names]

    txt = ""
    txt += " ".join(m_names)
    txt += "\n"
    txt += " ".join(b_names)
    txt += "\n\n"
    for i in range(num_r):
        txt += " ".join(m[i, :].astype(str).tolist())
        txt += "\n"

    if os.path.isfile(m_file):
        logging.warning("Overwriting file, '%s'" % m_file)
    with open(m_file, 'w') as f:
        f.write(txt)
    logging.info("Wrote matrix to %s" % m_file)


def tract_to_county(cen):
    """Convenience method to convert U.S. census tracts to county level.

    Parameters
    ----------
    cen : geopandas.GeoDataFrame
        A geo-data frame of U.S. census tracts (e.g., from :func:`get_census_geo`).

    Returns
    -------
    geopandas.GeoDataFrame
        The dissolved census tract data to U.S. county level;
        requires a new column, 'STCO', for unique state-county identification.

    Raises
    ------
    IndexError
        If the state-county column(s) are not found in the census data frame.
    """
    # Find the unique county column to dissolve the census tract
    d_col = 'STCO'
    if 'STCO' in cen.columns:
        pass
    elif 'STATEFP' in cen.columns and 'COUNTYFP' in cen.columns:
        logging.info("Creating a new state-county ID")
        cen['STCO'] = cen['STATEFP'] + cen['COUNTYFP']
    elif 'STATEFP' not in cen.columns or 'COUNTYFP' not in cen.columns:
        raise IndexError("Census data frame missing state and county columns!")

    logging.info("Dissolving tract features to county level...")
    county = cen.dissolve(by=d_col)
    logging.info("Done!")

    keep_cols = [
        'STATEFP', 'STUSPS', 'STATE_NAME', 'COUNTYFP', 'NAMELSADCO', 'geometry']
    drop_cols = [x for x in county.columns if x not in keep_cols]
    county = county.drop(columns=drop_cols)
    county = county.reset_index(drop=False)

    return county


def tract_to_state(cen):
    """Convenience method to convert U.S. census tract to states.

    Args:
        cen (geopandas.GeoDataFrame): A geo-data frame of U.S. census tracts
        (e.g., as returned by :func:`get_census_geo`)

    Raises:
        IndexError: If the data frame lacks a state ID column
        (e.g., 'STATEFP', 'STUSPS', or 'STATE_NAME')

    Returns:
        geopandas.GeoDataFrame: A dissolved geo-data frame of U.S. states.
    """
    # Find the unique state column to dissolve the census tract
    d_col = ''
    if 'STATEFP' in cen.columns:
        d_col = 'STATEFP'
    elif 'STUSPS' in cen.columns:
        d_col = 'STUSPS'
    elif 'STATE_NAME' in cen.columns:
        d_col = 'STATE_NAME'
    else:
        raise IndexError("Census data frame has not state column!")

    logging.info("Dissolving tract features to state level...")
    state = cen.dissolve(by=d_col)
    logging.info("Done!")

    keep_cols = ['STUSPS', 'STATE_NAME', 'geometry']
    drop_cols = [x for x in state.columns if x not in keep_cols]
    state = state.drop(columns=drop_cols)
    state = state.reset_index(drop=False)

    return state
