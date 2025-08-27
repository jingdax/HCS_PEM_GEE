
// ================== User Inputs ==================

var roi = /* your geometry */;

var training_2020 = /* FeatureCollection with features */;

var minMax_2020 = /* FeatureCollection holding min/max per band */;

var training_2020_GYH = /* normalized training FeatureCollection */;

var Variables = /* Image of predictor bands*/;

// ============== Image normalization ==============

function normalizeImage(image, minMaxDict) {
  var bandNames = image.bandNames();
  var normBands = bandNames.map(function(name) {
    name = ee.String(name);
    var band = image.select(name);
    var min = ee.Number(minMaxDict.get(name.cat('_min')));
    var max = ee.Number(minMaxDict.get(name.cat('_max')));
    return band.unitScale(min, max).rename(name);
  });

  return ee.ImageCollection.fromImages(normBands).toBands().rename(bandNames);
}

// ============== Runner ==============

var resultDict = {}

function runModelSet(year, stats, training, training_GYH, allLevels, ImageCollection) {
   
  allLevels.forEach(function(item) {
    
    var field = item.field;
    var labelName = item.labelName;
    var values = item.values;
    var fromList = item.remapFrom;
    var toList = item.remapTo;
    
    //Extracting stratified training samples
    
    var training_filtered = values.length > 0 ?
        training.filter(ee.Filter.inList(labelName, values)) :
        training;
    
    var training_GYH_filtered = values.length > 0 ?
        training_GYH.filter(ee.Filter.inList(labelName, values)) :
        training_GYH;
    
    //RF model training
    
    var RF = ee.Classifier.smileRandomForest({numberOfTrees: 200, seed: 42})
               .train({features: training_filtered, classProperty: labelName, inputProperties: Variables.bandNames()}).setOutputMode('MULTIPROBABILITY');
               
    
    //GTB model training
    
    var GTB = ee.Classifier.smileGradientTreeBoost({numberOfTrees: 160, shrinkage: 0.1, samplingRate: 0.75, maxNodes: 10, seed: 42})
                .train({features: training_filtered, classProperty: labelName, inputProperties: Variables.bandNames()}).setOutputMode('MULTIPROBABILITY');
    
    //SVM model training
    
    var SVM = ee.Classifier.libsvm({kernelType: 'RBF', gamma: 0.25, cost: 128})
                .train({features: training_GYH_filtered, classProperty: labelName, inputProperties: Variables.bandNames()}).setOutputMode('MULTIPROBABILITY');
    
    //Variables normalization
    
    var stats_feature = ee.Feature(stats.first()).toDictionary();
    
    var normalizedVariables = normalizeImage(Variables, stats_feature);
    
    //Classification
    
    var RF_pred =  Variables.classify(RF)
    var GTB_pred = Variables.classify(GTB)
    var SVM_pred = normalizedVariables.classify(SVM)
    
    //PEM
    
    var integrated = RF_pred.add(GTB_pred).add(SVM_pred).divide(3).arrayArgmax().arrayGet(0).remap(fromList, toList);
    
    resultDict[field] = integrated;

  })
  
  //Layered Merging
  
  var YRD_EM_50 =   resultDict['L1'].eq(0).remap([0, 1], [0, 50]);
  var YRD_EM_10 =   resultDict['L1'].eq(1).add(resultDict['L2A'].eq(3)).add(resultDict['L3'].eq(7)).add(resultDict['L4A'].eq(9)).eq(4).remap([0, 1], [0, 10]);
  var YRD_EM_20 =   resultDict['L1'].eq(1).add(resultDict['L2A'].eq(3)).add(resultDict['L3'].eq(7)).add(resultDict['L4A'].eq(11)).eq(4).remap([0, 1], [0, 20]);
  var YRD_EM_30 =   resultDict['L1'].eq(1).add(resultDict['L2A'].eq(3)).add(resultDict['L3'].eq(8)).add(resultDict['L4B'].eq(12)).eq(4).remap([0, 1], [0, 30]);
  var YRD_EM_412 =  resultDict['L1'].eq(1).add(resultDict['L2A'].eq(3)).add(resultDict['L3'].eq(8)).add(resultDict['L4B'].eq(13)).eq(4).remap([0, 1], [0, 412]);
  var YRD_EM_70 =   resultDict['L1'].eq(1).add(resultDict['L2A'].eq(4)).add(resultDict['L4C'].eq(14)).eq(3).remap([0, 1], [0, 70]);
  var YRD_EM_81 =   resultDict['L1'].eq(1).add(resultDict['L2A'].eq(4)).add(resultDict['L4C'].eq(15)).eq(3).remap([0, 1], [0, 81]);
  var YRD_EM_411 =  resultDict['L1'].eq(2).add(resultDict['L2B'].eq(5)).add(resultDict['L4D'].eq(16)).eq(3).remap([0, 1], [0, 411]);
  var YRD_EM_62 =   resultDict['L1'].eq(2).add(resultDict['L2B'].eq(5)).add(resultDict['L4D'].eq(17)).eq(3).remap([0, 1], [0, 62]);
  var YRD_EM_63 =   resultDict['L1'].eq(2).add(resultDict['L2B'].eq(6)).eq(2).remap([0, 1], [0, 63]);
  
  var YRD_EM = YRD_EM_50.add(YRD_EM_10).add(YRD_EM_20).add(YRD_EM_30).add(YRD_EM_412).add(YRD_EM_70).add(YRD_EM_81).add(YRD_EM_411)
               .add(YRD_EM_62).add(YRD_EM_63);
  
  Export.image.toDrive({
    image: YRD_EM, 
    scale: 30, 
    region: roi, 
    description: 'YRD_EM_' + year, 
    folder: "YRD_EM", 
    crs: "EPSG:4326", 
    maxPixels: 1e13
  });
  
}
  
//run

runModelSet('2020', minMax_2020, training_2020, training_2020_GYH, [
  
  {field: 'L1',  labelName: 'L1', values: [],          remapFrom: ee.List([0, 1, 2]), remapTo: ee.List([0, 1, 2])},
  {field: 'L2A', labelName: 'L2', values: [3, 4],      remapFrom: ee.List([0, 1]),    remapTo: ee.List([3, 4])},
  {field: 'L2B', labelName: 'L2', values: [5, 6],      remapFrom: ee.List([0, 1]),    remapTo: ee.List([5, 6])},
  {field: 'L3',  labelName: 'L3', values: [7, 8],      remapFrom: ee.List([0, 1]),    remapTo: ee.List([7, 8])},
  {field: 'L4A', labelName: 'L4', values: [9,  11],    remapFrom: ee.List([0, 1]),    remapTo: ee.List([9,  11])},
  {field: 'L4B', labelName: 'L4', values: [12, 13],    remapFrom: ee.List([0, 1]),    remapTo: ee.List([12, 13])},
  {field: 'L4C', labelName: 'L4', values: [14, 15],    remapFrom: ee.List([0, 1]),    remapTo: ee.List([14, 15])},
  {field: 'L4D', labelName: 'L4', values: [16, 17],    remapFrom: ee.List([0, 1]),    remapTo: ee.List([16, 17])},
  
  ],Variables)
  