/**
 * Stage 1: Data Collection and Preparation
 * Module: Meteorological Reanalysis Parameters (TMP, WS, Pre, ET)
 * Product: ECMWF ERA5-Land Monthly Aggregated (~11.1 km resolution)
 * Temporal Coverage: Annual and Monthly composites for 2025
 */

var region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq("country_na", "Iran"))
  .geometry();

var targetYear = 2025;

// Compute derived meteorological bands and convert units
var processERA5 = function(image) {
  var raw = image.select([
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_precipitation_sum",
    "total_evaporation_sum"
  ]);

  // 2 m air temperature (°C)
  var tmp = raw.select("temperature_2m").subtract(273.15).rename("TMP");

  // 10 m horizontal wind speed magnitude: sqrt(u^2 + v^2)
  var u = raw.select("u_component_of_wind_10m");
  var v = raw.select("v_component_of_wind_10m");
  var ws = u.pow(2).add(v.pow(2)).sqrt().rename("VS");

  // Accumulated precipitation and evapotranspiration
  var pre = raw.select("total_precipitation_sum").rename("Pre");
  var et = raw.select("total_evaporation_sum").rename("ET");

  return ee.Image.cat([tmp, ws, pre, et]).copyProperties(image, ["system:time_start"]);
};

var era5Collection = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
  .filterBounds(region)
  .filter(ee.Filter.calendarRange(targetYear, targetYear, "year"))
  .map(processERA5);

// 1. Annual Mean Composites
var annualEra5 = era5Collection.mean().clip(region);

var tmpVis = {
  min: -5.0,
  max: 40.0,
  palette: ["000080", "0000D9", "0080FF", "00FFFF", "80FF00", "FFFF00", "FFA400", "FF2500", "FF00FF"]
};

Map.centerObject(region, 5);
Map.addLayer(annualEra5.select("TMP"), tmpVis, "ERA5-Land Mean 2m Air Temperature (°C) " + targetYear);

var bands = [
  {band: "TMP", prefix: "Temperature"},
  {band: "VS",  prefix: "WindSpeed"},
  {band: "Pre", prefix: "Precipitation"},
  {band: "ET",  prefix: "Evaporation"}
];

bands.forEach(function(item) {
  Export.image.toDrive({
    image: annualEra5.select(item.band),
    description: item.prefix + "_ERA5_Yearly_" + targetYear + "_11km",
    folder: "SUHI_Iran_GEE",
    region: region,
    scale: 11132,
    maxPixels: 1e13,
    fileFormat: "GeoTIFF"
  });
});

// 2. Monthly Composites (Months 01 to 12)
for (var m = 1; m <= 12; m++) {
  var monthlyMean = era5Collection
    .filter(ee.Filter.calendarRange(m, m, "month"))
    .mean()
    .clip(region);

  var monthStr = (m < 10) ? "0" + m : m.toString();

  bands.forEach(function(item) {
    Export.image.toDrive({
      image: monthlyMean.select(item.band),
      description: item.prefix + "_ERA5_Monthly_" + targetYear + "_" + monthStr + "_11km",
      folder: "SUHI_Iran_GEE",
      region: region,
      scale: 11132,
      maxPixels: 1e13,
      fileFormat: "GeoTIFF"
    });
  });
}
