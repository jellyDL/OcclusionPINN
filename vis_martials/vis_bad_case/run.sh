python ../vis_occusion_heatmap/adjust_jaw.py -u data/case3-UpperJaw.ply -l data/case3_raw-LowerJaw.ply \
-d 0.3 -t-0 -sf 0 -p 0 -sl -0
python ../vis_final_result/vis_transparent_occlusion.py -u data/case3-UpperJaw.ply -l data/lower_shifted.ply -n test -z 1.2
