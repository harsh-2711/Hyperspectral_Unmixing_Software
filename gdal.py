from osgeo import gdal

dataset = gdal.Open("jasper.img", gdal.GA_ReadOnly)
print(dataset)