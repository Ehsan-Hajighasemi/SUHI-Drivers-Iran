# Stage 1: Satellite Remote Sensing & Gridded Data Extraction (Google Earth Engine)

This directory contains the JavaScript scripts executed in the **Google Earth Engine (GEE) Code Editor** to acquire and preprocess the national-scale raster layers across Iran prior to city-level buffer delineation and zonal statistics (Stage 2).

## Scripts Overview

| Script | Extracted Variables | Satellite / Reanalysis Product | Native Spatial Resolution |
| :--- | :--- | :--- | :--- |
| `01_extract_modis_lst.js` | Daytime & Nighttime $\text{LST}$ (Annual & 12 Months) | `MODIS/061/MYD11A1` (Aqua v6.1) | 1,000 m |
| `02_extract_landsat_ndvi.js` | $\text{NDVI}$ (Annual & 12 Months) | `LANDSAT/COMPOSITES/C02/T1_L2_32DAY_NDVI` | 30 m |
| `03_extract_era5_meteorology.js` | $\text{TMP}$, $\text{WS}$, $\text{Pre}$, $\text{ET}$ (Annual & 12 Months) | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | ~11,132 m |
| `04_extract_ghsl_morphology_pop.js` | $\text{BH}_{\text{mn}}$, $\text{BH}_{\text{std}}$, $\text{BS}_{\text{max}}$, $\text{BS}_{\text{std}}$, $\text{BV}_{\text{max}}$, $\text{POP}$ | `JRC/GHSL/P2023A` (`BUILT_H`, `BUILT_S`, `BUILT_V`, `POP`) | 100 m |
| `05_extract_dem_gdp_treecover.js` | $\text{DEM}$, $\text{GDP}$, $\text{FTC}$ | `USGS/GTOPO30`, Gridded GDP-HDI, `ESA/WorldCover/v200` | 927.67 m / 10 m |

## Usage

1. Open the [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. Copy and paste any of the `.js` scripts above into the editor and click **Run**.
3. By default, each script automatically clips exports to the national boundary of Iran using the `USDOS/LSIB_SIMPLE/2017` feature collection. You may optionally substitute `var region = table;` with your own uploaded study-area shapefile asset.
4. Navigate to the **Tasks** tab in the upper-right panel to initiate the GeoTIFF exports to Google Drive.
