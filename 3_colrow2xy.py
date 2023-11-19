import rasterio
from affine import Affine
# https://affine.readthedocs.io/en/latest/index.html

img1 = 'HyMap_Haib_Kornia.jpg'
img2 = 'WV3_Haib_Kornia.jpg'
imgout = 'HyMap_Haib_Kornia_coreg_WV3.tif'

ds1 = rasterio.open(img1)
ds2 = rasterio.open(img2)

# Using rasterio and affine
gt1 = ds1.get_transform()
gt2 = ds2.get_transform()
#print(gt1)
#print(gt2)
fwd1 = Affine.from_gdal(*gt1)
fwd2 = Affine.from_gdal(*gt2)

import csv
print("gdal_translate -co \"WORLDFILE=YES\" ", end='')
gcps = []
with open('gcps.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    headers = reader.fieldnames
    #print(headers)
    for row in reader:
        # col, row to x, y
        #x, y = fwd * (col, row)
        proj_x, proj_y = fwd2 * (int(row[headers[2]]),int(row[headers[3]]))
        gcps.append([int(row[headers[0]]),int(row[headers[1]]),round(proj_x,6),round(proj_y,6)])

# Sort and merge duplicates in list
# print(gcps)
from collections import defaultdict

def calculate_average(values):
    a_values = [item[0] for item in values]
    b_values = [item[1] for item in values]

    avg_a = sum(a_values) / len(a_values)
    avg_b = sum(b_values) / len(b_values)
    return [int(avg_a), int(avg_b), values[0][2], values[0][3]]

def group_and_average(data):
    grouped_data = defaultdict(list)

    for item in data:
        # Convert the inner list to a tuple to make it hashable
        tuple_item = tuple(item[2:])
        grouped_data[tuple_item].append(item)

    averages = [calculate_average(values) for values in grouped_data.values()]
    return averages

# Example data
data = [
    [115, 7, 772430.6968421012, 6824967.345454545],
    [195, 9, 773969.4972631513, 6824827.727272727],
    # ... 
]

# Calculate averages for unique integer tuples
avgs = group_and_average(gcps)
#print("Averages for each unique tuple:", averages)

# print unique gcps 
for i in range(len(avgs)):
    print("-gcp %d %d %f %f " %(avgs[i][0],avgs[i][1],avgs[i][2],avgs[i][3]), end='')

print(img1+" temp.jpg")

print("gdalwarp -overwrite -tps -t_srs \"EPSG:32733\" temp.jpg "+imgout)
# col, row to x, y
#x, y = fwd * (col, row)

# x, y to col, row
#rev = ~fwd
#col, row = ~rev * (x, y)

