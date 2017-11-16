ls *.tif>file_list.txt
gdalbuildvrt -te -180 -90 180 90 global_dem.vrt -input_file_list file_list.txt
