python vis_heatmap.py -l  data/test1_lower_open.ply   -n combined_upper_lower_open.png  --vmin -0.1  --vmax 1.0 --add_colorbar 0
python vis_heatmap.py -l  data/test1_lower_final.ply  -n combined_upper_lower_final.png --vmin -0.1  --vmax 1.0 --add_colorbar 1

python merge_images.py combined_upper_lower_open.png combined_upper_lower_final.png -o test1_final.png --spacing 100  -c "heatmap"