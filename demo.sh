python .\\demo\\topdown_demo_with_mmdet.py  projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth configs\\rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py work_dirs\\rtmpose-t_8xb1024-700e_body8-halpe26-256x192\\best_AUC_epoch_5.pth --input "C:\Users\jiriv\Downloads\VID_20230205_164423_HSR_120~2.mp4" --show --det-cat-id 0 --draw-bbox --device cpu

python .\\demo\\topdown_demo_with_mmdet.py  projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth configs\\rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth" --input "C:\Users\jiriv\Downloads\VID_20230205_164423_HSR_120~2.mp4" --show --det-cat-id 0 --draw-bbox --device cpu

python .\\demo\\topdown_demo_with_mmdet.py  projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth .\work_dirs\rtmpose-m_8xb256-420e_body8-256x192\rtmpose-m_8xb256-420e_body8-256x192.py .\work_dirs\rtmpose-m_8xb256-420e_body8-256x192\epoch4.pth       --input "C:\Users\jiriv\Videos\Starting and riding off rollers… shorts.mp4" --show --det-cat-id 0 --draw-bbox --device cpu --output-root fig

python .\\demo\\topdown_demo_with_mmdet.py 
 projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py 
 https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth 
 configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-384x288.py  
 https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-384x288-65e718c4_20230504.pth 
 --input "C:\Users\jiriv\OneDrive\bikefit videa\data\val\img\zlute_196.jpg"
 --show --det-cat-id 0 --device cpu --output-root fig

 python .\\demo\\topdown_demo_with_mmdet.py 
 projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py 
 https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth 
 configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-384x288.py 
 https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-384x288-89e6428b_20230605.pth
 --input "C:\Users\jiriv\OneDrive\bikefit videa\data\val\img\zlute_196.jpg"
 --show --det-cat-id 0 --device cpu --output-root fig

 python .\\demo\\topdown_demo_with_mmdet.py 
 projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py 
 https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth 
 configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb64-270e_coco-wholebody-256x192.py
 https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-coco-wholebody_pt-aic-coco_270e-256x192-6f206314_20230124.pth
 --input "C:\Users\jiriv\OneDrive\bikefit videa\data\val\img\zlute_196.jpg"
 --show --det-cat-id 0 --device cpu --output-root fig

python .\\demo\\topdown_demo_with_mmdet.py 
 projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py 
 https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth 
 configs/body_2d_keypoint/simcc/coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192.py 
 https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_vipnas-mbv3_8xb64-210e_coco-256x192-719f3489_20220922.pth 
 --input "C:\Users\jiriv\OneDrive\bikefit videa\data\val\img\zlute_196.jpg"
 --show --det-cat-id 0 --device cpu --output-root fig