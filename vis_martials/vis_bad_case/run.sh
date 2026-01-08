# python ../vis_occusion_heatmap/adjust_jaw.py -u data/case3-UpperJaw.ply -l data/case3_raw-LowerJaw.ply -d 0.3 -t-0 -sf 0 -p 0 -sl -0
# python ../vis_final_result/vis_transparent_occlusion.py -u data/case3-UpperJaw.ply -l data/lower_shifted.ply -n test -z 1.2


python ../vis_final_result/vis_transparent_occlusion.py -u data/case1-UpperJaw.ply -l data/case1_raw-LowerJaw.ply  -n case1_raw --vmin -0.2 --vmax 1.0 -z 0.8
python ../vis_final_result/vis_transparent_occlusion.py -u data/case1-UpperJaw.ply -l data/case1_opt-LowerJaw.ply  -n case1_opt --vmin -0.2 --vmax 1.0 -z 0.8
python ../vis_final_result/vis_transparent_occlusion.py -u data/case1-UpperJaw.ply -l data/case1_gt-LowerJaw.ply   -n case1_gt  --vmin -0.2 --vmax 1.0 -z 0.8

python ../vis_occusion_heatmap/merge_images.py view_front_case1_raw.png view_front_case1_opt.png view_front_case1_gt.png -o view_front_case1.png --spacing 200 --spacing-color "235,235,235" -c "Case 1"

python ../vis_final_result/vis_transparent_occlusion.py -u data/case2-UpperJaw.ply -l data/case2_raw-LowerJaw.ply  -n case2_raw --vmin -0.2 --vmax 1.0 -z 1.1
python ../vis_final_result/vis_transparent_occlusion.py -u data/case2-UpperJaw.ply -l data/case2_opt-LowerJaw.ply  -n case2_opt --vmin -0.2 --vmax 1.0 -z 1.1
python ../vis_final_result/vis_transparent_occlusion.py -u data/case2-UpperJaw.ply -l data/case2_gt-LowerJaw.ply   -n case2_gt  --vmin -0.2 --vmax 1.0 -z 1.1

python ../vis_occusion_heatmap/merge_images.py view_front_case2_raw.png view_front_case2_opt.png view_front_case2_gt.png -o view_front_case2.png --spacing 200 --spacing-color "235,235,235" -c "Case 2"

python ../vis_occusion_heatmap/merge_images.py view_front_case1.png view_front_case2.png  -o bad_case.png -d vertical -c "badcase" --spacing 50  --spacing-color "235,235,235"

python draw_ellipse_on_image.py -i bad_case.png -o bad_case_ellipse.png 