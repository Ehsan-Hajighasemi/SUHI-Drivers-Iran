/**
 * Stage 1: Data Collection and Preparation
 * Module: Surface Elevation (DEM), Per-Capita GDP, and Fractional Tree Cover (FTC)
 * Products:
 *   1. USGS GTOPO30 Global 30 Arc-Second Elevation (~927.67 m)
 *   2. Gridded GDP-HDI Per-Capita Total GDP (30 arc-seconds, ~927.67 m)
 *   3. ESA WorldCover 10 m v200 (Class 10: Tree Cover binary mask)
 */

var region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq("country_na", "Iran"))
  .geometry();

// 1. USGS GTOPO30 Digital Elevation Model
var gtopoDem = ee.Image("USGS/GTOPO30")
  .select("elevation")
  .clip(region);

// 2. Gridded Per-Capita GDP (latest available epoch)
var gdpMultiBand = ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/total_gdp_perCapita_1990_2020_30arcsec");
var latestBand = gdpMultiBand.bandNames().get(-1);
var gdpLatest = gdpMultiBand.select([latestBand]).clip(region);

// 3. ESA WorldCover v200 Tree Cover Binary Mask (1 = Tree cover [class 10], 0 = Other)
var worldcover = ee.ImageCollection("ESA/WorldCover/v200")
  .first()
  .select("Map")
  .clip(region);
var treeCover = worldcover.eq(10).rename("TreeCover");

Map.centerObject(region, 5);
Map.addLayer(gtopoDem, {min: 0, max: 3000, palette: ["0000ff", "00ffff", "ffff00", "ff0000", "ffffff"]}, "GTOPO30 DEM (m)");
Map.addLayer(treeCover, {min: 0, max: 1, palette: ["ffffff", "006400"]}, "ESA WorldCover Tree Cover (10 m)");

// Exports to Google Drive
Export.image.toDrive({
  image: gtopoDem,
  description: "DEM_GTOPO30_1km",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 927.67,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: gdpLatest,
  description: "GDP_Total_Latest_1km",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 927.67,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: treeCover,
  description: "TreeCover_ESAWorldCover2021_10m",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 10,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});
