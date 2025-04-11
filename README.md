# README
A "universal" spatial translation method from any given spatial extent to the U.S. census tract level.

This method is designed for disaggregation of numeric attributes from a larger spatial scale (e.g., state level) down to the census tract or county; it was not tested for upscaling.

In the current version, the starting spatial extents are predefined, but any extent with an appropriate spatial data file (e.g., shapefile or geoJSON) can be added to the code (with appropriate methods to handle the data acquisition and any necessary preprocessing, such as removing or renaming features).

The disaggregation methods are also predefined.
In the current version, two methods are available: equal weighting (all census tracts within the boundary of the larger spatial extent receive the same amount, 1/N), and areal weighting (each census tract receives a portion from the larger spatial extents in proportion to the fraction of coverage of each feature in the larger spatial extent that a census tract covers).


# Intended Application
 The intended application of this tool is for conversion of Life cycle assessment (LSA) inventories and impacts from various spatial extents to the census tract level.
 In this way, it can help to better assess community-level impacts of various projects.


# Instructions

## Creating the 'M' matrix
The first step in performing regionalization to the census tract or county level is to create the 'M' matrix, a 2-D numpy array with rows equal to the number of starting regions (e.g., 71 balancing authorities) and columns equal to the number of census tracts (e.g., 84,122) or counties, which contains the relevant conversion factors.
The 'M' matrices can be created using the `get_m` function.

```python
from regionalizer import get_m

M, codes, cen_codes = get_m(starting_extent='ST',
                            ending_extent = 'CT',
                            weighting='Eq',
                            state='US')`
```

-   `starting_extent` refers to the regional level on which data is given for conversion to the census tract (or county) level.
    Currently supports one of "BA" (balancing authority), "CB" (Coal basin), "NB" (natural gas basin), "ST" (state), "CO" (county), and "NS" (NERC sub-region).
    This can also be modified to include other spatial extents.
-   `ending_extent` refers to the regional level to which the data is converted.
    Currently supports one of "CT" (census tract) or "CO" (county).
    The code was developed to convert to the census tract level, but has been expanded to include county as well.
-   `Weighting` refers to the type of weighting applied. Currently supports either "A" (areal weighting, or impact proportional to area), or "Eq" (Equal weighting, impact equal for all census tracts in a given spatial extent).
-   `state` refers to the state for which census tracts are used.
    Choose by state (e.g., '54' is West Virginia).
    Note that these are strings and any number less than 10 needs a zero padding.
    Use 'US' for the entire United States.

Note that this may take some time to run, particularly when state='US', as a result of the volume of data being processed.
However, once the 'M' matrix is created, it can be reused, and subsequent calculations should be able to be performed fairly quickly.
The `get_m` function will automatically store the M matrix as a text document in the outputs folder, and retrieve it the next time the function is run with the same inputs.


## Regionalize using the the 'M' matrix
Once the 'M' matrix is created, it may be applied to convert data from the chosen spatial extent to the census tract or county level using the selected weighting method.
For an example of how this may be implemented see in the [Regionalization Notebook](./Regionalization-Notebook.ipynb#Testing).


# Limitations
There are several limitations associated with this tool in its current state, as outlined below.

1.  While balancing authorities are among the starting extents currently supported by the tool, it is important to note that these are not meant to be geographic regions, so there is uncertainty surrounding their exact boundaries.
2.  LCAs are often based on functional units, such as emissions/MWh.
    However, these are not suitable for regionalization.
    For instance, emissions/MWh units are already normalized based on electricity generation, which is a regional characteristic.
    Users should consider converting to total emissions per region before regionalizing, as shown in the Prepare the data section in the [Regionalization Notebook](./Regionalization-Notebook.ipynb#Testing).
3.  Some starting regions may not provide 100% coverage (e.g., NERC regions around the Florida peninsula, Louisiana bayou, New England coastline, have less coverage compared to US census tracts).
    If a census tract (or county) is partially uncovered, the areal weighting results may be impacted.
4.  In the case of the equal weighting method, edge effects may come into play, where spatial extents which lie on the edge of a census tract may be counted as overlapping.
    This is particularly notable for the case of balancing authorities, where a census tract may be bordered by several balancing authorities, which could lead to up to seven balancing authorities which are considered to overlap with a particular census tract.
5.  A space separated operator is used to store the list of unique identifiers associated with the starting spatial extents.
    Thus, these should not have any spaces in these identifiers, as words separated by spaces will be returned as separate codes.
    The `save_m` function attempts to address this issue by replacing spaces with underscores, however it may not always be clear how to map back to the original codes.
    Thus, it is recommended to ensure that there are no spaces in the unique identifier codes being used.
6.  There is some inconsistency among sources regarding the codes used for the balancing authorities GRIDFORCE SOUTH and GRIDFORCE ENERGY MANAGEMENT, LLC.
    In this case, 'GRIS' is used for GRIDFORCE SOUTH and 'GRID' is used for GRIDFORCE ENERGY MANAGEMENT, LLC.
