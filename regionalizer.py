#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# regionalizer.py

##############################################################################
# REQUIRED MODULES
##############################################################################
import logging
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import requests


##############################################################################
# DOCUMENTATION
##############################################################################
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
    2025-06-02
"""
__all__ = [
    "BA_YEAR",
    "CENSUS_YEAR",
    "DATA_DIR",
    "EOPTS",
    "MATRIX_DIR",
    "PCS",
    "ROPTS",
    "SHAPES_DIR",
    "SPATIAL_LV",
    "STATE_FILTER",
    "WOPTS",
    "area_weighting",
    "calculate_m",
    "census_weighting",
    "correct_cb_geo_names",
    "correct_nb_geo_names",
    "correct_ns_geo_names",
    "download_file",
    "equal_weighting",
    "filter_valid_ba",
    "get_ba_geo",
    "get_cb_geo",
    "get_census_geo",
    "get_county_geo",
    "get_em_geo",
    "get_io_value",
    "get_logger",
    "get_m",
    "get_nb_geo",
    "get_nercsub_geo",
    "get_region_col",
    "get_state_geo",
    "print_progress",
    "read_m",
    "run_unit_test",
    "save_m",
    "show_end_extent",
    "show_start_extent",
    "tract_to_county",
    "tract_to_state",
]


##############################################################################
# GLOBALS
##############################################################################
BA_YEAR_MAX = 2024
BA_YEAR_MIN = 2000
BA_YEAR = 2020
'''int : Data vintage for Energy Atlas balancing authorities.'''
CENSUS_YEAR = 2020
'''int : Data vintage for US Census Bureau's census shapefile (2020--2023).'''
DATA_DIR = "inputs"
'''str: Local directory for storing data files.'''
SHAPES_DIR = os.path.join(DATA_DIR, "shapes")
'''str : Local directory for storing GeoJSON and shapefiles.'''
MATRIX_DIR = os.path.join(DATA_DIR, "matrices")
'''str : Local directory for storing M files.'''
IO_PR_LIMIT = True
'''bool : Whether to limit the progress bar message to a threshold.'''
IO_PR_FRACTION = 0.01
'''bool : The fraction of progress bar messages to show (larger == less).'''
PCS = ('esri', 102009)
'''tuple: Geopandas CRS info for North America Lambert Conformal Conic.'''
SPATIAL_LV = pd.Series(
    [5, 4, 4, 3, 3, 3, 2, 1],
    index=["EM", "NS", "BA", "CB", "NB", "ST", "CO", "CT"]
)
'''pandas.Series : Spatial/regional hierarchy levels.'''
SPATIAL_LV_DESC = pd.Series(
    ["Electricity Market Module Regions (EIA)",
     "NERC Subregions (EIA)",
     "Balancing Authorities (U.S. Energy Atlas)",
     "Coal Basins (EIA)",
     "Natural Gas Basins (EPA)",
     "U.S. States (Census Bureau)",
     "U.S. Counties (Census Bureau)",
     "U.S. Census Tracts (Census Bureau)"],
    index=["EM", "NS", "BA", "CB", "NB", "ST", "CO", "CT"]
)
'''pandas.Series : Spatial/regional hierarchy level descriptions.'''
STATE_FILTER = ['PR', 'MP', 'AS', 'GU', 'VI']
'''list : A list of state and territory codes to remove from census tract.'''
ROPTS = ['BA', 'NB', 'CB', 'ST','NS','CO', 'EM']
'''list : Abbreviations of available starting region options.'''
EOPTS = ['CT','BA', 'NB', 'CB', 'ST','NS','CO', ]
'''list : Abbreviations of available ending region options.'''
WOPTS = ['A', 'Eq','Cen']
'''list : Abbreviations of available weighting options.'''
_loggers = {}
'''dict : A dictionary for storing logging instances.'''


##############################################################################
# CLASSES
##############################################################################
class Regionalizer(object):
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Global Variables
    # ////////////////////////////////////////////////////////////////////////
    BA_YEAR = 2020
    '''int : Data vintage for Energy Atlas balancing authorities.'''
    CENSUS_YEAR = 2020
    '''int : Data vintage for US Census Bureau's data (2020--2023).'''
    DATA_DIR = "inputs"
    '''str: Local directory for storing data files.'''
    EOPTS = ['CT','BA', 'NB', 'CB', 'ST','NS','CO', ]
    '''list : Abbreviations of available ending region options.'''
    IO_PR_LIMIT = True
    '''bool : Whether to limit the progress bar message to a threshold.'''
    IO_PR_FRACTION = 0.01
    '''bool : The fraction of progress bar messages to show (larger == less).'''
    ROPTS = ['BA', 'NB', 'CB', 'ST','NS','CO', 'EM']
    '''list : Abbreviations of available starting region options.'''
    STATE_FILTER = ['PR', 'MP', 'AS', 'GU', 'VI']
    '''list : List of state and territory codes to remove from census data.'''
    SHAPES_DIR = os.path.join(DATA_DIR, "shapes")
    '''str : Local directory for storing GeoJSON and shapefile files.'''
    MATRIX_DIR = os.path.join(DATA_DIR, "matrices")
    '''str : Local directory for storing M files.'''
    PCS = ('esri', 102009)
    '''tuple: Geopandas CRS info for North America Lambert Conformal Conic.'''
    SPATIAL_LV = pd.Series(
        [5, 4, 4, 3, 3, 3, 2, 1],
        index=["EM", "NS", "BA", "CB", "NB", "ST", "CO", "CT"]
    )
    '''pandas.Series : Spatial/regional hierarchy levels.'''
    SPATIAL_LV_DESC = pd.Series(
        ["Electricity Market Module Regions (EIA)",
        "NERC Subregions (EIA)",
        "Balancing Authorities (U.S. Energy Atlas)",
        "Coal Basins (EIA)",
        "Natural Gas Basins (EPA)",
        "U.S. States (Census Bureau)",
        "U.S. Counties (Census Bureau)",
        "U.S. Census Tracts (Census Bureau)"],
        index=["EM", "NS", "BA", "CB", "NB", "ST", "CO", "CT"]
    )
    '''pandas.Series : Spatial/regional hierarchy level descriptions.'''
    WOPTS = ['A', 'Eq','Cen']
    '''list : Abbreviations of available weighting options.'''

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Initialization
    # ////////////////////////////////////////////////////////////////////////
    def __init__(self):
        # Create a class-level logger
        self.log = logging.getLogger("Regionalizer")

        # Turn off IO limited if not in Jupyter environment.
        _is_jupyter = is_running_jupyter()
        if not _is_jupyter:
            self.IO_PR_LIMIT = False

        # Set default options
        self._show_progress = True

        # Store the user's options.
        self._ba_year = None
        self._census_year = None
        self._end_extent = None
        self._start_extent = None

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Property Definitions
    # ////////////////////////////////////////////////////////////////////////
    @property
    def ba_year(self):
        if self._ba_year is None:
            return self.BA_YEAR
        else:
            return self._ba_year

    @ba_year.setter
    def ba_year(self, val):
        if not isinstance(val, int):
            try:
                my_val = int(val)
            except ValueError:
                raise TypeError("the BA year must be an integer!")
            else:
                val = my_val
        self._ba_year = val

    @property
    def census_year(self):
        if self._census_year is None:
            return self.CENSUS_YEAR
        else:
            return self._census_year

    @census_year.setter
    def census_year(self, val):
        if not isinstance(val, int):
            try:
                my_val = int(val)
            except ValueError:
                raise TypeError("the census year must be an integer!")
            else:
                val = my_val
        self._census_year = val

    @property
    def end_extent(self):
        # Alias
        return self._end_extent

    @property
    def ending_region(self):
        # Alias
        return self._end_extent

    @property
    def start_extent(self):
        # Alias
        return self._start_extent

    @property
    def starting_region(self):
        # Alias
        return self._start_extent

    @property
    def ending_options(self):
        return self.SPATIAL_LV_DESC[self.SPATIAL_LV_DESC.index.isin(self.EOPTS)]

    @property
    def starting_options(self):
        return self.SPATIAL_LV_DESC[self.SPATIAL_LV_DESC.index.isin(self.ROPTS)]

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # Class Method Definitions
    # ////////////////////////////////////////////////////////////////////////
    def show_end_extents(self):
        """Display names and abbreviations for ending extent options."""
        print("Abbr\tDescription")
        print("----\t-----------")
        for idx, val in self.ending_options.items():
            print("%s\t%s" % (idx, val))

    def show_start_extents(self):
        """Display names and abbreviations for starting extent options."""
        print("Abbr\tDescription")
        print("----\t-----------")
        for idx, val in self.starting_options.items():
            print("%s\t%s" % (idx, val))


##############################################################################
# FUNCTIONS
##############################################################################
def area_weighting(ee, se, name_column, ee_name_column, overlap=True):
    """Calculates conversion factors from a starting spatial extent to the
    ending extent (e.g. census tract) level using an area weighting method (i.e., coefficients
    proportional to shared area).

    Parameters
    ----------
    ee : geopandas.geodataframe.GeoDataFrame
        A geopandas data frame with rows for ending extent (e.g., census
        tracts or counties), with column, ee_name_column, containing the
        unique feature identifier (e.g., GEOID or state and county FIPS ID),
        which should be projected into a coordinate reference system with
        linear unit (e.g., meter).
    se : geopandas.geodataframe.GeoDataFrame
        A geopandas data frame with rows for starting extent features,
        with column, 'name_column', containing a unique identifier for each
        starting extent feature. The data frame should be in the same
        reference coordinate system as the ending extent.
    name_column : str
        The name of the column in se with unique identifiers for the starting
        extent  (e.g., U.S. state abbreviation).
    ee_name_column : str
        The name of the column in ee with unique identifiers for ending
        extent  (e.g., county GEOID).

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame containing the conversion factors from the
        spatial extents in starting extent to the census tracts or counties
        using the area weighting method.
    """
    # Re-calculate their areas (square kilometers)
    logging.info("Calculating polygon areas")
    se['SE_KM2'] = se['geometry'].area / 10**6
    ee['CEN_KM2'] = ee['geometry'].area / 10**6

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

    if overlap==True:
        # Create a total column of overlap areas that contribute to the
        # starting region. Note that if this is greater than the starting
        # extent total area, it indicates overlaps in the starting region,
        # and can lead to over-supply to ending regions.
        # NOTE: you could calculate these two areas, do a check for greater
        # than and automatically do this overlap compensation.
        df_reg_totals = df.groupby(name_column).agg({'AREA_KM2': 'sum'})
        df['TOT_AREA_KM2'] = df.index.get_level_values(name_column).map(
            df_reg_totals['AREA_KM2'])
        df['VALUE'] = df['AREA_KM2'] / df['TOT_AREA_KM2']
    else:
        df['VALUE'] = df['AREA_KM2'] / df['SE_KM2']

    return (df)


def calculate_m(starting_extent,
                ending_extent,
                weighting,
                state='US',
                ba_year=BA_YEAR,
                census_year=CENSUS_YEAR):
    """Calculate the mapping matrix, M, for translating a starting spatial
    extent (e.g., balancing authority, coal basin, natural gas basin, or state)
    to and ending spatial extent (e.g., U.S census tracts or counties) using
    either areal or equal weighting.

    Parameters
    ----------
    starting_extent : str
        Spatial extent your input data is associated with.
        Currently supports one of the following,

        -   "BA", Balancing authority
        -   "CB", Coal basin
        -   "NB", Natural gas basin
        -   "ST", State
        -   "CO", County
        -   "NS", NERC sub-region
        -   "EM", EIA Electricity Market Module Regions

    ending_extent : str
        Spatial extent your output data is associated with.
        Currently supports one of the following,

        -   "CT", Census tract (community level)
        -   "CO", County
        -   "BA", Balancing authority
            (BA can only be used if the starting_extent is a larger region,
            such as EM)

    weighting : str
        Weighting method to use. Choose from,

        -   "A", Areal weighting or for impact proportional to area
        -   "Eq", Equal weighting or for impact equal for all ending regions

    state : str (optional)
        When ending extent is 'state', 'county' or 'census', only the features
        for the given state are returned; use 'US' as the state for all states.
        Defaults to 'US'. For state-level codes (e.g., '54' is West Virginia),
        see https://www2.census.gov/geo/docs/reference/codes2020/national_state2020.txt
        Note that these are strings and any number less than 10 needs
        zero padding.

    ba_year : int (optional)
        If starting_extent or ending_extent is balancing authority, this
        determines the year to associate with balancing authorities; it
        filters BAs based on their operational and retirement status.
        Defaults to global, BA_YEAR.

    census_year : int (optional)
        If the starting or ending region is 'state', 'county', or 'tract',
        then this determines which year from the US census bureau to use.
        Defaults to global, CENSUS_YEAR.

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
    ValueError
        In the event that starting region is 'smaller' than ending region.
    """
    # Address translation/disaggregation region hierarchy.
    if SPATIAL_LV[starting_extent] < SPATIAL_LV[ending_extent]:
        raise ValueError(
            "Please choose an ending extent that is the same size as or "
            "smaller than the starting extent (EM>BA=NS>CB=NB=ST>CO>CT)"
        )

    # Manage ending extent
    ee_name_column = get_region_col(ending_extent)
    if ending_extent == "CT":
        ee = get_census_geo(census_year, state, make_prj=True)
    elif ending_extent == "CO":
        ee = get_county_geo(census_year, state, make_prj=True)
    elif ending_extent == "BA":
        ee = get_ba_geo(ba_year, correct_names=True, make_prj=True)
    elif ending_extent == "CB":
        ee = get_cb_geo(correct_names=True, make_prj=True)
    elif ending_extent == "ST":
        ee = get_state_geo(census_year, state, make_prj=True)
    elif ending_extent == "NS":
        ee = get_nercsub_geo(correct_names=True, make_prj=True)
    else:
        raise ValueError("Ending region, '%s', not handled!" % ending_extent)

    # Manage starting extent
    name_column = get_region_col(starting_extent)
    if starting_extent == "BA":
        se = get_ba_geo(ba_year, correct_names=True, make_prj=True)
    elif starting_extent == "CB":
        se = get_cb_geo(correct_names=True, make_prj=True)
    elif starting_extent == "ST":
        se = get_state_geo(census_year, state, make_prj=True)
    elif starting_extent == "CO":
        se = get_county_geo(census_year, state, make_prj=True)
    elif starting_extent == "NB":
        se = get_nb_geo(correct_names=True, make_prj=True)
    elif starting_extent == "NS":
        se = get_nercsub_geo(correct_names=True, make_prj=True)
    elif starting_extent == "EM":
        se = get_em_geo(make_prj=True)
    else:
        raise ValueError("Starting region, '%s', unhandled!" % starting_extent)

    # Add check for name column in data frames
    if ee_name_column not in ee.columns:
        raise IndexError(
            "Something went wrong with the name column in ending extent!")
    if name_column not in se.columns:
        raise IndexError(
            "Something went wrong with the name column in starting extent!")

    # HOTFIX: rename starting region unique identifier in ending region if
    # found; causes problems with overlay later on [250429; TWD]
    if name_column in ee.columns:
        new_name = "%s_ee" % name_column
        ee = ee.rename(columns={name_column: new_name})

    # Save the lengths and name lists (for later use)
    num_se = len(se)
    num_ee = len(ee)

    # The lists of starting and ending region identifiers
    se_list = se[name_column].tolist()
    ee_list = ee[ee_name_column].tolist()

    # Perform overlay
    if weighting == 'A':
        df = area_weighting(ee, se, name_column, ee_name_column, overlap=True)
    elif weighting == 'Eq':
        df = equal_weighting(ee, se, name_column, ee_name_column)
    elif weighting == 'Cen':
        df = census_weighting(ee, se, name_column, ee_name_column)

    # Initialize M matrix
    logging.info("Calculating conversion matrix")
    M = np.zeros(shape=(num_se, num_ee))

    # Get totals and IO-message governor (for Jupyter Notebooks)
    total = num_se * num_ee
    governor = get_io_value(total)

    # Place values into M
    for n in range(num_se):
        se_name = se_list[n]
        for m in range(num_ee):
            ee_name = ee_list[m]
            val = 0
            try:
                val = df.loc[(se_name, ee_name), 'VALUE']
            except KeyError:
                pass
            else:
                M[n,m] = val

            # Progress bar; probably need 4 decimal places
            cur_step = n*num_ee + m + 1
            if IO_PR_LIMIT and cur_step % governor == 0:
                print_progress(cur_step, total, "", 'Complete', 4)
            elif not IO_PR_LIMIT or cur_step == total:
                print_progress(cur_step, total, "", 'Complete', 4)

    # Warn about potential vacancies and over/under allocations
    logging.info("Checking spatial representation of conversion matrix")
    for n in range(num_se):
        se_name = se_list[n]
        if M[n,:].sum() == 0:
            logging.warning(
                "Zero representation for starting region(s), '%s'" % se_name)
        elif M[n,:].sum() > 1.0 + 1e9:
            logging.warning("Over allocation from '%s'" % se_name)
        elif M[n,:].sum() < 1.0 - 1e9:
            logging.warning("Under allocation from '%s'" % se_name)
    for m in range(num_ee):
        ee_name = ee_list[m]
        if M[:,m].sum() == 0:
            if IO_PR_LIMIT:
                # For small starting regions and large ending regions, there
                # can be a lot of messages that crash Jupyter Notebook.
                logging.debug(
                    "Zero representation for ending region(s), '%s'" % ee_name)
            else:
                logging.warning(
                    "Zero representation for ending region(s), '%s'" % ee_name)

    # Correct ending extent names back to strings for writing to file.
    ee_list = [x for x in ee[ee_name_column]]
    return (M, se_list, ee_list)


def census_weighting(ee, se, name_column, ee_name_column,
                     census_year=CENSUS_YEAR):
    """Calculate conversion factors from a starting spatial extent to the
    ending extent level using a census weighting method (i.e., coefficients
    proportional to overlapping census tract). This can be used as a
    proxy for population weighting.

    Parameters
    ----------
    ee : geopandas.geodataframe.GeoDataFrame
        A geopandas data frame with rows for ending extent (e.g., census
        tracts or counties), with column, ee_name_column, containing the
        unique feature identifier (e.g., GEOID or state and county FIPS ID),
        which should be projected into a coordinate reference system with
        linear unit (e.g., meter).
    se : geopandas.geodataframe.GeoDataFrame
        A geopandas data frame with rows for starting extent features,
        with column, 'name_column', containing a unique identifier for each
        starting extent feature. The data frame should be in the same
        reference coordinate system as the ending extent.
    name_column : str
        The name of the column in se with unique identifiers for the starting
        extent  (e.g., U.S. state abbreviation).
    ee_name_column : str
        The name of the column in ee with unique identifiers for ending
        extent  (e.g., county GEOID).
    census_year : int
        The year to reference census tract data.
        Defaults to global, CENSUS_YEAR.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame containing the conversion factors from the
        spatial extents in starting extent to the census tracts or counties
        using the population (by proxy of census tracts) weighting method.
    """
    census = get_census_geo(census_year, 'US', make_prj=True)

    logging.info("Calculating polygon areas")
    census['CEN_KM2'] = census['geometry'].area / 10**6

    logging.info("Intersecting data frames...")
    eb = gpd.overlay(se, ee, how='intersection')

    # Calculate the area of each intersection
    logging.info("Computing intersection areas")
    eb['AREA_KM2'] = eb['geometry'].area / 10**6
    eb['Intersection'] = eb[name_column] + ' ' + eb[ee_name_column]

    # HOTFIX: avoid intersecting layers with the same unique identifier found
    # in spatial extents. [250429; TWD]
    if name_column in census.columns:
        new_name = "%s_cen" % name_column
        census = census.rename(columns={name_column: new_name})
    if ee_name_column in census.columns:
        new_name = "%s_cen" % ee_name_column
        census = census.rename(columns={ee_name_column: new_name})

    # Overlay census tracts
    eb_census = gpd.overlay(eb, census, how='intersection')

    # Calculate the intersection areas
    eb_census['CEN_OVERLAP_AREA_KM2'] = eb_census['geometry'].area / 10**6

    # Find area fraction (CEN_OVERLAP_AREA_KM2/CEN_KM2) for each overlap
    eb_census['AREA_FRACTION'] = eb_census['CEN_OVERLAP_AREA_KM2']/eb_census['CEN_KM2']

    df = eb_census.groupby(
        by=[name_column, ee_name_column]
    ).agg({
        'AREA_FRACTION': 'sum'
    })
    df_reg_totals = eb_census.groupby(
        by=[name_column]
    ).agg({
        'AREA_FRACTION': 'sum'
    })
    # NOTE: this uses the areas of census tracts as a proxy for population
    df['TOT_AREA_KM2'] = df.index.get_level_values(name_column).map(
        df_reg_totals['AREA_FRACTION'])

    df['VALUE'] = df['AREA_FRACTION'] / df['TOT_AREA_KM2']

    return df


def check_output_dir(out_dir):
    """Helper method to ensure a directory exists.

    If a given directory does not exist, this method attempts to create it.

    Parameters
    ----------
    out_dir : str
        A path to a directory.

    Returns
    -------
    bool
        Whether the directory exists.

    Notes
    -----
    Source: ElectricityLCI (https://github.com/USEPA/ElectricityLCI)
    """
    if not os.path.isdir(out_dir):
        try:
            # Start with super mkdir
            os.makedirs(out_dir)
        except:
            logging.warning("Failed to create folder %s!" % out_dir)
            try:
                # Revert to simple mkdir
                os.mkdir(out_dir)
            except:
                logging.error("Could not create folder, %s" % out_dir)
            else:
                logging.info("Created %s" % out_dir)
        else:
            logging.info("Created %s" % out_dir)

    return os.path.isdir(out_dir)


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
        'CA-MX US': 'CA-MX',
        'NEW ENGLAND': 'NE',
        'NEW YORK': 'NY',
        'MRO US': 'MRO'
    }
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


def equal_weighting(ee, se, name_column, ee_name_column):
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
    ee_name_column : str
        The name of the column in ee with unique identifiers.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame containing the conversion factors from the
        spatial extents in starting extent to the census tracts using the
        equal weighting method.
    """
    logging.info("Intersecting data frames...")
    eb = gpd.overlay(se, ee, how='intersection')

    logging.info("Counting intersections")
    eb['count'] = eb.groupby(name_column)[name_column].transform('count')
    df = eb.groupby(
        by=[name_column, ee_name_column]
    )[['count']].agg({'count': 'sum'})

    logging.info("Normalizing counts")
    df['VALUE'] = 1/df['count'].values

    return df


def filter_valid_ba(gdf, year):
    """Filter a data frame to get rows valid for a given year.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame.
    year : int
        The year to filter against.

    Returns
    -------
    geopandas.GeoDataFrame
        The filtered GeoDataFrame.
    """
    # Convert the year to datetime objects for comparison
    year_start = gpd.pd.to_datetime(f'{year}-01-01')
    year_end = gpd.pd.to_datetime(f'{year}-12-31')

    # Filter for rows where start date is before or equal to the given year
    started = gdf['Op_Date'] <= year_end

    # Filter for rows where RetireDate is NaT or after or equal to the
    # given year
    not_retired_or_valid_retirement = (
        gdf['Ret_Date'].isna() | (gdf['Ret_Date'] >= year_start)
    )

    # Combine the filters
    filtered_gdf = gdf.loc[started & not_retired_or_valid_retirement]

    return filtered_gdf


def get_ba_geo(year, correct_names=False, make_prj=False):
    """Create a geospatial data frame of U.S. balancing authorities from
    U.S. Energy Atlas using their API service,
    https://atlas.eia.gov/datasets/09550598922b429ca9f06b9a067257bd_255/explore

    Parameters
    ----------
    year : int
        The year to associated balancing authorities; filters operational
        and retirement status.
    correct_names : bool, optional
        Convert 'EIAcode' column to 'BA_CODES' and 'EIAname' to 'BA_NAME'
        for convenience. Defaults to false.
    make_prj : bool, optional
        If true, convert to standard projected coordinate system.
        Defaults to false.

    Returns
    -------
    geopandas.DataFrame
        A spatial data frame with the following columns:

        - OBJECTID (int)
        - EIAcode / BA_CODE (str)
        - EIAname / BA_NAME (str)
        - EIAregion (str)
        - Op_Date (datetime64 [ns])
        - Ret_Date (datetime64 [ns])
        - Year (str)
        - NAICS_CODE (str)
        - NAICS_DESC (str)
        - HIFLDid (str)
        - HIFLDname (str)
        - HIFLDsource (str)
        - HIFLDweb (str)
        - Shape__Area (float64)
        - Shape__Length (float64)
        - geometry (geometry)
    """
    eia_atlas_url = (
        "https://services7.arcgis.com"
        "/FGr1D95XCGALKXqM/arcgis/rest/services/Balancing_Authorities"
        "/FeatureServer/255/query?outFields=*&where=1%3D1&f=geojson"
    )
    f_name = "balancing_authorities.geojson"
    f_path = os.path.join(SHAPES_DIR, f_name)
    if not os.path.isfile(f_path) and check_output_dir(SHAPES_DIR):
        download_file(eia_atlas_url, f_path)
    ba_geo = gpd.read_file(f_path)

    # Convert integers to datetime [ms].
    # NOTE: 'coerce' will result in NaT for invalid datetimes.
    ba_geo['Ret_Date'] = gpd.pd.to_datetime(
        ba_geo['Ret_Date'], unit="ms", errors='coerce')
    ba_geo['Op_Date'] = gpd.pd.to_datetime(
        ba_geo['Op_Date'], unit="ms", errors='coerce')

    # Filter by year (for BAs not operational or retired)
    ba_geo = filter_valid_ba(ba_geo, year)

    if correct_names:
        ba_geo = ba_geo.rename(columns={
            'EIAname': 'BA_NAME',
            'EIAcode': 'BA_CODE'
        })
    if make_prj:
        logging.info("Projecting balancing authority map")
        ba_geo = ba_geo.to_crs(PCS)

    return ba_geo


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
    ba_path = os.path.join(SHAPES_DIR, ba_file)

    # Use existing file if available:
    if not os.path.isfile(ba_path) and check_output_dir(SHAPES_DIR):
        download_file(ba_api_url, ba_path)

    # Read GeoJSON and correct BA area names (if requested)
    gdf = gpd.read_file(ba_path, layer='CBMbasins_resources_2006')

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
    The Cartographic Boundary Files are published by the U.S. Census Bureau.
    Census tracts are areas of land within the United States that
    are home to roughly 4,000 people each.
    They always follow county lines, and may also follow boundaries
    such as municipal lines, rivers, and roads. Due to the small
    size of census tracts, this is a large shapefile, containing
    85,187 features in the 2020 census.

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
        census tracts at 500k resolution (1:500,000), with columns,

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
    cen_path = os.path.join(SHAPES_DIR, cen_file)

    # Handle missing census data files.
    if not os.path.isfile(cen_path) and check_output_dir(SHAPES_DIR):
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


def get_county_geo(year, region, make_prj=False):
    """Read US county shapefile for a given year and region.

    Notes
    -----

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
        counties at 500k resolution, with columns,

        -   STATEFP (str), two digit state ID (zero padded)
        -   COUNTYFP (str), three digit county ID (zero padded)
        -   STUSPS (str), two-character state abbreviation
        -   STATE_NAME (str), state name
        -   STCO (str), unique state-county identification, found by combining
            STATEFP and COUNTYFP columns

    Raises
    ------
    OSError
        The zipped shapefile is not publicly accessible. This error raises
        when the local file is not found.
    """
    # Create the census file name, (e.g., "cb_2020_us_county_500k.zip")
    if isinstance(region, int):
        region = "%02d" % region
    region = region.lower()
    # HOTFIX: match the data resolution with census tracts.
    cen_file = "cb_%d_us_county_500k.zip"  % (year)
    cen_path = os.path.join(SHAPES_DIR, cen_file)

    # Handle missing census data files.
    if not os.path.isfile(cen_path) and check_output_dir(SHAPES_DIR):
        logging.info("Downloading the county file, %s" % cen_file)
        cen_url = "https://www2.census.gov/geo/tiger/GENZ%d/shp/%s" % (
            year, cen_file)
        _worked = download_file(cen_url, cen_path)
        if _worked:
            logging.info("County file downloaded!")

    logging.info("Reading county shapefile")
    gdf = gpd.read_file(cen_path)

    # Remove non-states from data frame
    gdf = gdf.query("STUSPS not in @STATE_FILTER")

    if make_prj:
        logging.info("Projecting county map")
        gdf = gdf.to_crs(PCS)

    # Find the unique county column to dissolve the county
    if 'STCO' in gdf.columns:
        pass
    elif 'STATEFP' in gdf.columns and 'COUNTYFP' in gdf.columns:
        logging.info("Creating a new state-county ID")
        gdf['STCO'] = gdf['STATEFP'] + gdf['COUNTYFP']
    elif 'STATEFP' not in gdf.columns or 'COUNTYFP' not in gdf.columns:
        raise IndexError("county data frame missing state and county columns!")

    keep_cols = [
        'STCO',
        'STATEFP',
        'STUSPS',
        'STATE_NAME',
        'COUNTYFP',
        'NAMELSADCO',
        'geometry'
    ]
    drop_cols = [x for x in gdf.columns if x not in keep_cols]
    gdf = gdf.drop(columns=drop_cols)
    if region != 'us':
        gdf = gdf[gdf['STATEFP'] == region]

    return gdf


def get_em_geo(make_prj=False):
    """Read EIA AEO electricity market module region shapefile.

    Notes
    -----
    The GIS shapefile for EIA's Electricity Market Module (EMM) regions were
    used to generate the map that EIA published. These shapes are not used
    directly in the AEO model or input data preparation. The map and shapefile
    were created during AEO2020 when it first went to 25 regions and are meant
    to represent the markets where EIA assigns the capacity and sales, and the
    service territories associated with those markets.

    The shapefile was delivered upon request to EIA on December 2, 2022.
    Available on EDX (2025-03-13):
    https://edx.netl.doe.gov/resource/7536c4db-e25b-4e48-8ac2-7da6f4e26da1/download

    Available online at: https://www.eia.gov/outlooks/aeo/additional_docs.php.

    Parameters
    ----------
    make_prj : bool, optional
        Whether to convert the coordinate system to PCS.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A geospatial data frame of polygon areas representing the U.S.
        electricity market module regions (EMMRs).

    Raises
    ------
    OSError
        The zipped shapefile is not publicly accessible.
        This error raises when the local file is not found.
    """

    em_file = "EMM_GIS_shapefile.zip"
    em_path = os.path.join(SHAPES_DIR, em_file)

    # Handle missing census data files.
    if not os.path.isfile(em_path) and check_output_dir(SHAPES_DIR):
        logging.info("Downloading the census file, %s" % em_file)
        em_url = (
            "https://www.eia.gov/outlooks/aeo/images/zip/EMM_GIS_shapefile.zip"
        )
        _worked = download_file(em_url, em_path)
        if _worked:
            logging.info("The EIA AEO2020 market region shapefile downloaded!")

    gdf = gpd.read_file(em_path)

    if make_prj:
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
        Depends on global parameter, `IO_PR_FRACTION`.

    Examples
    --------
    >>> get_io_value(123456789)
    123500
    """
    return max(1, int(round(max_val*IO_PR_FRACTION, -2)))


def get_logger(name='root'):
    """Convenience function for retrieving loggers by name."""
    if name not in _loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
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


def get_m(start_extent, end_extent, weighting, state='US', ba_year=BA_YEAR):
    """Return the mapping matrix and name lists for starting extent to census
    (or county) translation.

    Parameters
    ----------
    starting_extent : str
        Spatial extent your input data is associated with.
        Currently supports one of the following,

        -   "BA", Balancing authority
        -   "CB", Coal basin
        -   "NB", Natural gas basin
        -   "ST", State
        -   "CO", County
        -   "NS", NERC sub-region
        -   "EM", EIA Electricity Market Module Regions

    ending_extent : str
        Spatial extent your output data is associated with.
        Currently supports one of the following,

        -   "CT", Census tract (community level)
        -   "CO", County
        -   "BA", Balancing authority
            (BA can only be used if the starting_extent is a larger region,
            such as EM)

    weighting : str
        The weighting method.
        Available options include:

        - 'A' areal weighting
        - 'Eq' equal weighting
        - 'Cen' population weighting

    state : str (optional)
        The region code for U.S. census tracts.
        Choose by state (e.g., '54' is West Virginia);
        note that these are strings and any number less than 10 needs
        a zero padding.
        Use 'US' for all U.S. census tracts.
        Defaults to 'US'.
    ba_year : int (if starting_extent or ending_extent = 'BA')
        The year to associated balancing authorities; filters operational
        and retirement status.

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
    if start_extent.lower() not in [x.lower() for x in ROPTS]:
        raise ValueError(
            "Starting region option, '%s', is not available." % start_extent)
    if end_extent.lower() not in [x.lower() for x in EOPTS]:
        raise ValueError(
            "Ending region option, '%s', is not available." % end_extent)
    if weighting.lower() not in [x.lower() for x in WOPTS]:
        raise ValueError(
            "Weighting option, '%s', is not available." % weighting)

    logging.info(
        "Getting matrix, M, for %s region to %s region using %s weighting." % (
            start_extent, end_extent, weighting
        )
    )

    # The m.txt file stores the conversion matrix.
    # It's location is defined here, but if it doesn't exist,
    # then the code will create the matrix and save it to the file.
    map_file_us_ct = os.path.join(
        MATRIX_DIR,
        "m_" + start_extent + "_" + weighting + ".txt"
    )

    map_file_ct = os.path.join(
        MATRIX_DIR,
        "m_" + start_extent + "_" + state + "_" + weighting + ".txt"
    )

    map_file = os.path.join(
        MATRIX_DIR,
        "m_" + start_extent + "_" + end_extent + "_" + state + "_" + weighting + ".txt"
    )

    # If the matrix file exists, read it; otherwise, create it.
    if os.path.isfile(map_file):
        logging.info("Reading mapping file")
        M, ba_list, cen_list = read_m(map_file)
    elif (
            os.path.isfile(map_file_us_ct)
            & (state == 'US')
            & (end_extent == 'CT')):
        logging.info("Reading mapping file")
        M, ba_list, cen_list = read_m(map_file_us_ct)
    elif os.path.isfile(map_file_ct) & (end_extent == 'CT'):
        logging.info("Reading mapping file")
        M, ba_list, cen_list = read_m(map_file_ct)
    else:
        logging.info("Creating mapping matrix")
        M, ba_list, cen_list = calculate_m(
            start_extent,
            end_extent,
            weighting,
            state,
            ba_year
        )
        if check_output_dir(MATRIX_DIR):
            save_m(M, ba_list, cen_list, map_file)

    return (M, ba_list, cen_list)


def get_nb_geo(correct_names=False, filter_basins=True, make_prj=False):
    """Create a geospatial data frame for U.S. natural gas basins.

    Notes
    -----
    Source: EPA's Office of Air and Radiation (OAR)
    and Office of Atmospheric Protection (OAP)
    Environmental Dataset Gateway (EDG)
    (https://edg.epa.gov/data/Public/OAR/OAP/)

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
    _path = os.path.join(SHAPES_DIR, _file)

    # Use existing file if available:
    if not os.path.isfile(_path) and check_output_dir(SHAPES_DIR):
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
    """Create a geospatial data frame for North American Electric
    Reliability Corporation (NERC) subregions.

    Run this method once to download a local copy of the GeoJSON.
    Subsequent runs of this method attempt to read the local file rather
    than re-download the file. The file name is "nerc_subregions.geojson" and
    is saved locally (e.g., see global, SHAPES_DIR).

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
    ba_path = os.path.join(SHAPES_DIR, ba_file)

    # Use existing file if available:
    if not os.path.isfile(ba_path) and check_output_dir(SHAPES_DIR):
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


def get_region_col(region_abbr):
    """Return the column name associated with the unique values for a given
    region.

    Parameters
    ----------
    region_abbr : str
        One of the valid region abbreviations (e.g., 'BA', 'CB', 'CO', 'CT',
        'EM', 'NB', 'NS', 'ST').

    Returns
    -------
    str
        Column name for unique values in the geodataframe associated with the
        given region.

    Raises
    ------
    ValueError
        If region abbreviation is not recognized.
    """
    if region_abbr == "BA":
        name_column = 'BA_CODE'
    elif region_abbr == "CB":
        name_column = 'basin'
    elif region_abbr == "CO":
        name_column = "STCO"
    elif region_abbr == "CT":
        name_column = "GEOID"
    elif region_abbr == "EM":
        name_column = "eGrid_Reg"
    elif region_abbr == "NB":
        name_column = "BASIN_CODE"
    elif region_abbr == "NS":
        name_column = "SUBNAME_CORRECTED"
    elif region_abbr == "ST":
        name_column = "STUSPS"
    else:
        raise ValueError("Region, '%s', unrecognized!" % region_abbr)

    return name_column


def get_state_geo(year, region, make_prj=False):
    """Read US state region shapefile for a given year and region.

    Notes
    -----
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
        census tracts at 500k resolution, with columns,

        - STATEFP (str), two digit state ID (zero padded)
        - STUSPS (str), two-character state abbreviation

    Raises
    ------
    OSError
        The zipped shapefile is not publicly accessible. This error raises
        when the local file is not found.
    """
    # Create the census file name, (e.g., "cb_2020_us_state_500k.zip")
    if isinstance(region, int):
        region = "%02d" % region
    region = region.lower()
    gdf_file = "cb_%d_%s_state_500k.zip"  % (year, region)
    gdf_path = os.path.join(SHAPES_DIR, gdf_file)

    # Handle missing census data files.
    if not os.path.isfile(gdf_path) and check_output_dir(SHAPES_DIR):
        logging.info("Downloading the census file, %s" % gdf_file)
        gdf_url = "https://www2.census.gov/geo/tiger/GENZ%d/shp/%s" % (
            year, gdf_file)
        _worked = download_file(gdf_url, gdf_path)
        if _worked:
            logging.info("state file downloaded!")

    logging.info("Reading state shapefile")
    gdf = gpd.read_file(gdf_path)

    # Remove non-states from data frame
    gdf = gdf.query("STUSPS not in @STATE_FILTER")

    if make_prj:
        logging.info("Projecting state map")
        gdf = gdf.to_crs(PCS)

    keep_cols = ['STATEFP', 'STUSPS', 'STATE_NAME', 'geometry']
    drop_cols = [x for x in gdf.columns if x not in keep_cols]
    if region != 'us':
        gdf = gdf[gdf['STATEFP'] == region]
    gdf = gdf.drop(columns=drop_cols)
    return gdf


def is_running_jupyter():
    """A helper method to determine if user is in a Jupyter environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # Thanks, G. Bezerra! https://stackoverflow.com/a/39662359
        if shell.__class__.__name__ != 'ZMQInteractiveShell':
            return False
        return True
    except ImportError:
        return False


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


def run_unit_test(matrix,             # M
                  end_df,             # ending region data frame
                  end_region_codes,   # from get_m
                  weight_method,      # 'area' or 'equal'
                  match_col,          # match col in ending region data frame
                  results,            # created outside this method
                  n):                 # row index of results
    # Get the sizing of the arrays
    rows, cols = matrix.shape
    if cols != len(end_region_codes):
        raise IndexError(
            "The number of rows does not match number of ending regions!")

    # Create the unit input array
    b = np.ones(rows)
    b = np.array(b)
    b = np.reshape(b, (1, rows))

    # The translation
    c = np.dot(b, matrix)

    c_dict = {
        match_col: [],
        'Num_SEs': [],
    }
    for i in range(cols):
        gid = end_region_codes[i]  # pull the Geoid
        val = c[0, i]              # pull the input amount
        c_dict[match_col].append(gid)
        c_dict['Num_SEs'].append(val)

    # HOTFIX: don't overwrite the input data frame
    df = pd.merge(end_df, pd.DataFrame(c_dict), how='left', on=match_col)

    weight_method = weight_method.lower()
    if weight_method == 'area' or weight_method == 'a':
        tot_col = 'Total Output Areal Weighting'
        diff_col = 'Areal Difference'
    elif weight_method == 'equal' or weight_method == 'eq':
        tot_col = 'Total Output Equal Weighting'
        diff_col = 'Equal Difference'
    else:
        raise ValueError("Weighting method not recognized!")

    results.loc[n, 'Total input'] = df['Num_SEs'].sum()
    results.loc[n, tot_col] = b.sum()
    results.loc[n, diff_col] = df['Num_SEs'].sum() - b.sum()

    return results


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


def show_end_extents():
    """Display names and abbreviations for ending extent options."""
    print("Abbr\tDescription")
    print("----\t-----------")
    for idx, val in SPATIAL_LV_DESC[SPATIAL_LV_DESC.index.isin(EOPTS)].items():
        print("%s\t%s" % (idx, val))


def show_start_extents():
    """Display names and abbreviations for starting extent options."""
    print("Abbr\tDescription")
    print("----\t-----------")
    for idx, val in SPATIAL_LV_DESC[SPATIAL_LV_DESC.index.isin(ROPTS)].items():
        print("%s\t%s" % (idx, val))


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
