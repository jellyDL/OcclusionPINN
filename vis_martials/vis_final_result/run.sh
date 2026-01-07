python vis_transparent_occlusion.py -u data_penetrate/case1-UpperJaw.stl -l data_penetrate/case1_raw-LowerJaw.stl -n case1_raw
python vis_transparent_occlusion.py -u data_penetrate/case1-UpperJaw.stl -l data_penetrate/case1_opt-LowerJaw.stl -n case1_opt
python vis_transparent_occlusion.py -u data_penetrate/case1-UpperJaw.stl -l data_penetrate/case1_gt-LowerJaw.stl -n case1_gt
python ../vis_occusion_heatmap/merge_images.py view_front_case1_raw.png view_front_case1_opt.png view_front_case1_gt.png -o view_front_case1.png --spacing 200 --spacing-color "235,235,235"
#
python vis_transparent_occlusion.py -u data_penetrate/case2-UpperJaw.ply -l data_penetrate/case2_raw-LowerJaw.ply -n case2_raw
python vis_transparent_occlusion.py -u data_penetrate/case2-UpperJaw.ply -l data_penetrate/case2_opt-LowerJaw.ply -n case2_opt
python vis_transparent_occlusion.py -u data_penetrate/case2-UpperJaw.ply -l data_penetrate/case2_gt-LowerJaw.ply -n case2_gt
python ../vis_occusion_heatmap/merge_images.py view_front_case2_raw.png view_front_case2_opt.png view_front_case2_gt.png -o view_front_case2.png --spacing 200 --spacing-color "235,235,235"
#
python vis_transparent_occlusion.py -u data_penetrate/case3-UpperJaw.ply -l data_penetrate/case3_raw-LowerJaw.ply -n case3_raw
python vis_transparent_occlusion.py -u data_penetrate/case3-UpperJaw.ply -l data_penetrate/case3_opt-LowerJaw.ply -n case3_opt
python vis_transparent_occlusion.py -u data_penetrate/case3-UpperJaw.ply -l data_penetrate/case3_gt-LowerJaw.ply -n case3_gt
python ../vis_occusion_heatmap/merge_images.py view_front_case3_raw.png view_front_case3_opt.png view_front_case3_gt.png -o view_front_case3.png --spacing 200 --spacing-color "235,235,235"
#
python vis_transparent_occlusion.py -u data_penetrate/case4-UpperJaw.ply -l data_penetrate/case4_raw-LowerJaw.ply -n case4_raw
python vis_transparent_occlusion.py -u data_penetrate/case4-UpperJaw.ply -l data_penetrate/case4_opt-LowerJaw.ply -n case4_opt
python vis_transparent_occlusion.py -u data_penetrate/case4-UpperJaw.ply -l data_penetrate/case4_gt-LowerJaw.ply -n case4_gt
python ../vis_occusion_heatmap/merge_images.py view_front_case4_raw.png view_front_case4_opt.png view_front_case4_gt.png -o view_front_case4.png --spacing 200 --spacing-color "235,235,235"
#
python ../vis_occusion_heatmap/merge_images.py view_front_case3.png view_front_case2.png view_front_case1.png view_front_case4.png  -o view_front_case.png -d vertical -c "T" --spacing 50  --spacing-color "235,235,235"

python draw_ellipse_on_image.py -i view_front_case.png -o view_front_case_final.png