/**
 * Stage 1: Data Collection and Preparation
 * Module: Daytime and Nighttime Land Surface Temperature (LST)
 * Product: MODIS Aqua MYD11A1 Version 6.1 (1,000 m resolution)
 * Temporal Coverage: Annual and Monthly composites for 2025
 */

// Define study area boundary (defaults to national boundary of Iran; replace with custom feature collection if needed)
var region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq("country_na", "Iran"))
  .geometry();

var targetYear = 2025;

// Apply scale factor (0.02) and convert LST bands from Kelvin to degrees Celsius
var toCelsius = function(image) {
  var lstCelsius = image.select(["LST_Day_1km", "LST_Night_1km"])
    .multiply(0.02)
    .subtract(273.15);
  return lstCelsius.copyProperties(image, ["system:time_start"]);
};

var lstCollection = ee.ImageCollection("MODIS/061/MYD11A1")
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(targetYear, targetYear, "year"))
  .map(toCelsius);

// 1. Annual Mean Composites (Daytime and Nighttime)
var annualMean = lstCollection.mean().clip(region);

var lstVis = {
  min: 10,
  max: 50,
  palette: [
    "040274", "0502a3", "0502ce", "0602ff", "307ef3", "30c8e2",
    "3be285", "86e26f", "b5e22e", "fff705", "ffb613", "ff6e08",
    "ff0000", "c21301", "911003"
  ]
};

Map.centerObject(region, 5);
Map.addLayer(annualMean.select("LST_Day_1km"), lstVis, "MODIS Aqua LST Day " + targetYear);

Export.image.toDrive({
  image: annualMean.select("LST_Day_1km"),
  description: "LST_Aqua_Yearly_Day_" + targetYear + "_1km",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 1000,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

Export.image.toDrive({
  image: annualMean.select("LST_Night_1km"),
  description: "LST_Aqua_Yearly_Night_" + targetYear + "_1km",
  folder: "SUHI_Iran_GEE",
  region: region,
  scale: 1000,
  maxPixels: 1e13,
  fileFormat: "GeoTIFF"
});

// 2. Monthly Mean Composites (Months 01 to 12)
for (var m = 1; m <= 12; m++) {
  var monthlyMean = lstCollection
    .filter(ee.Filter.calendarRange(m, m, "month"))
    .mean()
    .clip(region);

  var monthStr = (m < 10) ? "0" + m : m.toString();

  Export.image.toDrive({
    image: monthlyMean.select("LST_Day_1km"),
    description: "LST_Aqua_Monthly_Day_" + targetYear + "_" + monthStr + "_1km",
    folder: "SUHI_Iran_GEE",
    region: region,
    scale: 1000,
    maxPixels: 1e13,
    fileFormat: "GeoTIFF"
  });

  Export.image.toDrive({
    image: monthlyMean.select("LST_Night_1km"),
    description: "LST_Aqua_Monthly_Night_" + targetYear + "_" + monthStr + "_1km",
    folder: "SUHI_Iran_GEE",
    region: region,
    scale: 1000,
    maxPixels: 1e13,
    fileFormat: "GeoTIFF"
  });
}
