/**
 * Stage 1: Data Collection and Preparation
 * Module: Normalized Difference Vegetation Index (NDVI)
 * Product: Landsat Collection 2 Tier 1 Level 2 32-Day NDVI Composite (30 m resolution)
 * Temporal Coverage: Annual and Monthly composites for 2025
 */

var region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq("country_na", "Iran"))
  .geometry();

var targetYear = 2025;

var ndviCollection = ee.ImageCollection("LANDSAT/COMPOSITES/C02/T1_L2_32DAY_NDVI")
  .filterBounds(region)
  .filterDate(targetYear + "-01-01", (targetYear + 1) + "-01-01")
  .select("NDVI");

// 1. Annual Mean NDVI Composite
var annualNdvi = ndviCollection.mean().clip(region);

var ndviVis = {
  min: -0.1,
  max: 0.8,
  palette: ["ffffff", "ce7e45", "df923d", "f1b555", "fcd163", "99b718", "74a901", "3e8601", "056201"]
};

Map.centerObject(region, 5);
Map.addLayer(annualNdvi, ndviVis, "Landsat Annual NDVI " + targetYear);

Export.image.toDrive({
  image: annualNdvi,
  description: "NDVI_Landsat_Annual_" + targetYear + "_30m",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 30,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

// 2. Monthly Mean NDVI Composites (Months 01 to 12)
for (var m = 1; m <= 12; m++) {
  var startDate = ee.Date.fromYMD(targetYear, m, 1);
  var endDate = startDate.advance(1, "month");

  var monthlyNdvi = ndviCollection
    .filterDate(startDate, endDate)
    .mean()
    .clip(region);

  var monthStr = (m < 10) ? "0" + m : m.toString();

  Export.image.toDrive({
    image: monthlyNdvi,
    description: "NDVI_Landsat_Monthly_" + targetYear + "_" + monthStr + "_30m",
    folder: "SUHI_Iran_GEE",
    region: region,
    scale: 30,
    maxPixels: 1e13,
    fileFormat: "GeoTIFF"
  });
}
