python .\\demo\\topdown_demo_with_mmdet.py  projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth configs\\rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py work_dirs\\rtmpose-t_8xb1024-700e_body8-halpe26-256x192\\best_AUC_epoch_5.pth --input "C:\Users\jiriv\Downloads\VID_20230205_164423_HSR_120~2.mp4" --show --det-cat-id 0 --draw-bbox --device cpu

python .\\demo\\topdown_demo_with_mmdet.py  projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth configs\\rtmpose-t_8xb1024-700e_body8-halpe26-256x192.py "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.pth" --input "C:\Users\jiriv\Downloads\VID_20230205_164423_HSR_120~2.mp4" --show --det-cat-id 0 --draw-bbox --device cpu

python .\\demo\\topdown_demo_with_mmdet.py  projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth .\work_dirs\rtmpose-m_8xb256-420e_body8-256x192\rtmpose-m_8xb256-420e_body8-256x192.py .\work_dirs\rtmpose-m_8xb256-420e_body8-256x192\epoch4.pth       --input "C:\Users\jiriv\Videos\Starting and riding off rollers… shorts.mp4" --show --det-cat-id 0 --draw-bbox --device cpu