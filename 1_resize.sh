# If you have more RAM set the larger image resolution
# WV3 image
cols=15821
rows=8776
# HyMap image
cols=3706
rows=1924
# S2 image
cols=950
rows=528
gdal_translate -co "WORLDFILE=YES" -outsize $cols $rows Namibia/EMIT_Haib_resize.tif EMIT_Haib_Kornia.jpg 
#gdal_translate -co "WORLDFILE=YES" -outsize $cols $rows Namibia/S2_Haib_resize.tif S2_Haib_Kornia.jpg 
gdal_translate -co "WORLDFILE=YES" -outsize $cols $rows Namibia/HyMap_Haib_resized.tif HyMap_Haib_Kornia.jpg 
gdal_translate -co "WORLDFILE=YES" -outsize $cols $rows Namibia/WV3_Haib_resize.tif WV3_Haib_Kornia.jpg 
