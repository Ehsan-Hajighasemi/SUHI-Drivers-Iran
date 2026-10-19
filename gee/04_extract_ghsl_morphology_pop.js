/**
 * Stage 1: Data Collection and Preparation
 * Module: 3D Urban Morphology (Building Height, Surface, Volume) and Population
 * Products: JRC GHSL P2023A (GHS_BUILT_H, GHS_BUILT_S, GHS_BUILT_V, GHS_POP) at 100 m resolution
 */

var region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq("country_na", "Iran"))
  .geometry();

var targetEpoch = 2025;

// 1. GHSL Building Height (static 2018 epoch, 100 m)
var builtHeight = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_H")
  .filterBounds(region)
  .select("built_height")
  .first()
  .clip(region);

// 2. GHSL Built-Up Surface Area (m^2 per 100 m grid cell, 2025 epoch)
var builtSurface = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(targetEpoch, targetEpoch, "year"))
  .select("built_surface")
  .first()
  .clip(region);

// 3. GHSL Built-Up Volume (m^3 per 100 m grid cell, 2025 epoch)
var builtVolume = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_V")
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(targetEpoch, targetEpoch, "year"))
  .select("built_volume_total")
  .first()
  .clip(region);

// 4. GHSL Residential Population Count (inhabitants per 100 m grid cell, 2025 epoch)
var population = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP")
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(targetEpoch, targetEpoch, "year"))
  .select("population_count")
  .first()
  .clip(region);

Map.centerObject(region, 5);
Map.addLayer(builtHeight, {min: 0, max: 30, palette: ["000000", "3B008D", "CC0066", "FF6600", "FFFF00"]}, "GHSL Building Height (m)");
Map.addLayer(population, {min: 0, max: 100, palette: ["000004", "3B0F70", "DE4968", "FE9F6D", "FCFDBF"]}, "GHSL Population " + targetEpoch);

// Export GeoTIFF layers at native 100 m resolution
Export.image.toDrive({
  image: builtHeight,
  description: "Built_Height_GHSL_2018_100m",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 100,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: builtSurface,
  description: "Built_Surface_GHSL_" + targetEpoch + "_100m",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 100,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: builtVolume,
  description: "Built_Volume_GHSL_" + targetEpoch + "_100m",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 100,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: population,
  description: "Population_GHSL_" + targetEpoch + "_100m",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 100,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});
