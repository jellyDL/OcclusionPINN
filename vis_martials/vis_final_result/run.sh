# python ../vis_occusion_heatmap/adjust_jaw.py -u data_penetrate/case5-UpperJaw.stl -l data_penetrate/case5_raw-LowerJaw.stl -d 0.10 -t 0.05 -p 0.6 -sf -0.15 -sl -0.15
# python vis_transparent_occlusion.py -u data_penetrate/case5-UpperJaw.stl -l data_penetrate/lower_shifted.stl -n test --vmin -0.2 --vmax 1.0


python vis_transparent_occlusion.py -u data_penetrate/case1-UpperJaw.stl -l data_penetrate/case1_raw-LowerJaw.stl -n case1_raw --vmin -0.1 --vmax 1.0
python vis_transparent_occlusion.py -u data_penetrate/case1-UpperJaw.stl -l data_penetrate/case1_opt-LowerJaw.stl -n case1_opt --vmin -0.1 --vmax 1.0
python vis_transparent_occlusion.py -u data_penetrate/case1-UpperJaw.stl -l data_penetrate/case1_gt-LowerJaw.stl  -n case1_gt  --vmin -0.1 --vmax 1.0
python ../vis_occusion_heatmap/merge_images.py view_front_case1_raw.png view_front_case1_opt.png view_front_case1_gt.png -o view_front_case1.png --spacing 200 --spacing-color "235,235,235" -c "Case 3"

python vis_transparent_occlusion.py -u data_penetrate/case2-UpperJaw.ply -l data_penetrate/case2_raw-LowerJaw.ply -n case2_raw --vmin -0.2 --vmax 1.0
python vis_transparent_occlusion.py -u data_penetrate/case2-UpperJaw.ply -l data_penetrate/case2_opt-LowerJaw.ply -n case2_opt --vmin -0.2 --vmax 1.0
python vis_transparent_occlusion.py -u data_penetrate/case2-UpperJaw.ply -l data_penetrate/case2_gt-LowerJaw.ply  -n case2_gt  --vmin -0.2 --vmax 1.0
python ../vis_occusion_heatmap/merge_images.py view_front_case2_raw.png view_front_case2_opt.png view_front_case2_gt.png -o view_front_case2.png --spacing 200 --spacing-color "235,235,235" -c "Case 2"

python vis_transparent_occlusion.py -u data_penetrate/case3-UpperJaw.stl -l data_penetrate/case3_raw-LowerJaw.stl -n case3_raw --vmin -0.2 --vmax 1.0
python vis_transparent_occlusion.py -u data_penetrate/case3-UpperJaw.stl -l data_penetrate/case3_opt-LowerJaw.stl -n case3_opt --vmin -0.2 --vmax 1.0
python vis_transparent_occlusion.py -u data_penetrate/case3-UpperJaw.stl -l data_penetrate/case3_gt-LowerJaw.stl  -n case3_gt  --vmin -0.2 --vmax 1.0
python ../vis_occusion_heatmap/merge_images.py view_front_case3_raw.png view_front_case3_opt.png view_front_case3_gt.png -o view_front_case3.png --spacing 200 --spacing-color "235,235,235" -c "Case 1"

python vis_transparent_occlusion.py -u data_penetrate/case4-UpperJaw.ply -l data_penetrate/case4_raw-LowerJaw.ply -n case4_raw --vmin -0.2 --vmax 0.7 -z 0.82
python vis_transparent_occlusion.py -u data_penetrate/case4-UpperJaw.ply -l data_penetrate/case4_opt-LowerJaw.ply -n case4_opt --vmin -0.2 --vmax 1.0 -z 0.82
python vis_transparent_occlusion.py -u data_penetrate/case4-UpperJaw.ply -l data_penetrate/case4_gt-LowerJaw.ply  -n case4_gt  --vmin -0.2 --vmax 1.0 -z 0.82
python ../vis_occusion_heatmap/merge_images.py view_front_case4_raw.png view_front_case4_opt.png view_front_case4_gt.png -o view_front_case4.png --spacing 200 --spacing-color "235,235,235" -c "Case 4"


python ../vis_occusion_heatmap/merge_images.py view_front_case3.png view_front_case2.png view_front_case1.png view_front_case4.png  -o view_front_case.png -d vertical -c "final" --spacing 50  --spacing-color "235,235,235"
python draw_ellipse_on_image.py -i view_front_case.png -o result_compare.png 


# python image_compress.py -i result_compare.png -o result_compare-c.png -q 40
