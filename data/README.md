# Dataset Structure and Variable Definitions

The analytical pipeline operates on a unified city-level tabular dataset (`Final_Database_corrected.csv`) covering **1,109 Iranian cities** stratified into three UNEP climatic zones: **Arid** (480 cities), **Semi-arid** (373 cities), and **Humid** (256 cities)[cite: 2, 7].

> **Note on Urban–Suburban Contrast ($\Delta$):** Variables prefixed with `Δ` represent the difference between the mean value within the urban boundary and the mean value within the corresponding suburban reference ring (merged 10th–12th concentric buffer rings, each equal in area to half the city footprint, with water bodies masked out at 20 m resolution):
> $$\Delta X = X_{\text{urban}} - X_{\text{suburban}}$$[cite: 11]

## 1. Identifiers and Spatial Metadata
| Column | Description |
| :--- | :--- |
| `City_ID` | Unique numerical identifier for each of the 1,109 cities |
| `long`, `lat` | Geographic coordinates (longitude and latitude in decimal degrees) |
| `Climate_Class` | UNEP climate zone classification (`Arid`, `Semi-arid`, `Humid`)[cite: 7] |

## 2. Target Variables (10 Outcomes)
Daytime and nighttime Surface Urban Heat Island (SUHI) intensity ($^\circ\text{C}$) derived from MODIS MYD11A1 (v6.1) 1,000 m Land Surface Temperature (LST) for 2025 across five temporal scales (`Annual`, `Winter`, `Spring`, `Summer`, `Autumn`)[cite: 10]:
* **Daytime:** `SUHI_Annual_Day`, `SUHI_Winter_Day`, `SUHI_Spring_Day`, `SUHI_Summer_Day`, `SUHI_Autumn_Day`
* **Nighttime:** `SUHI_Annual_Night`, `SUHI_Winter_Night`, `SUHI_Spring_Night`, `SUHI_Summer_Night`, `SUHI_Autumn_Night`

## 3. Explanatory Drivers (19 Variables per Outcome)

### Hydro-Climatic, Biophysical, and Topographic Drivers
| Column Name(s) | Manuscript Symbol | Description & Data Source |
| :--- | :--- | :--- |
| `ΔNDVI_<Season>` | $\Delta\text{NDVI}$ | Urban–suburban difference in Normalized Difference Vegetation Index (Landsat C2 L2, 30 m)[cite: 10, 12] |
| `TMP_<Season>` | $\text{TMP}$ | Mean 2 m air temperature in the urban area (ERA5-Land, ~11 km)[cite: 10, 12] |
| `VS_<Season>` | $\text{WS}$ | Mean 10 m horizontal wind speed in the urban area (ERA5-Land, ~11 km)[cite: 10, 12] |
| `Pre_<Season>` | $\text{Pre}$ | Mean accumulated precipitation in the urban area (ERA5-Land, ~11 km)[cite: 10, 12] |
| `ET_<Season>` | $\text{ET}$ | Mean accumulated evapotranspiration in the urban area (ERA5-Land, ~11 km)[cite: 10, 12] |
| `ΔDEM` | $\Delta\text{DEM}$ | Urban–suburban difference in mean surface elevation (GTOPO30, 927 m)[cite: 10, 12] |

### Anthropogenic and Socioeconomic Drivers
| Column Name | Manuscript Symbol | Description & Data Source |
| :--- | :--- | :--- |
| `ΔPOP` | $\Delta\text{POP}$ | Urban–suburban difference in population density (GHSL, 100 m)[cite: 10, 12] |
| `ΔFTC` | $\Delta\text{FTC}$ | Urban–suburban difference in fractional tree cover percentage (ESA WorldCover, 10 m)[cite: 10, 12] |
| `ΔGDP` | $\Delta\text{GDP}$ | Urban–suburban difference in per-capita Gross Domestic Product (Gridded GDP-HDI, 30 arc-sec)[cite: 10, 12] |

### 2D Urban Morphology (FRAGSTATS Landscape Metrics)
| Column Name | Manuscript Symbol | Description |
| :--- | :--- | :--- |
| `PLAND` | $\text{PLAND}$ | Percentage of urban landscape occupied by built-up patches[cite: 12, 13] |
| `PD` | $\text{PD}$ | Built-up patch density (number of patches per unit area)[cite: 13] |
| `ED` | $\text{ED}$ | Total edge length of built-up patches per unit area[cite: 13] |
| `LSI` | $\text{LSI}$ | Landscape Shape Index measuring horizontal geometric dispersion/compactness[cite: 13] |
| `FRAC` | $\text{FRAC}$ | Fractal Dimension Index measuring patch boundary complexity[cite: 13] |

### 3D Urban Morphology (GHS-BUILT, 100 m)
| Column Name | Manuscript Symbol | Description |
| :--- | :--- | :--- |
| `BH_mn` | $\text{BH}_{\text{mn}}$ | Mean building height within the urban area[cite: 10, 13] |
| `BH_std` | $\text{BH}_{\text{std}}$ | Standard deviation of building heights within the urban area[cite: 10, 13] |
| `BS_max` | $\text{BS}_{\text{max}}$ | Maximum building footprint area recorded in the urban area[cite: 10, 13] |
| `BS_std` | $\text{BS}_{\text{std}}$ | Standard deviation of building footprint areas in the urban area[cite: 10, 13] |
| `BV_max` | $\text{BV}_{\text{max}}$ | Maximum building volume recorded in the urban area[cite: 10, 13] |
