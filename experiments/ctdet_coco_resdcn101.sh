cd src
# train
python main.py ctdet --exp_id coco_resdcn101 --arch resdcn_101 --batch_size 96 --master_batch 5 --lr 3.75e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16
# test
python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_resdcn101 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
python main.py multi_pose --exp_id res50_dcn_3x_usnavy512 --dataset coco_hp --arch resdcn_50 --batch_size 24 --master_batch 12 --lr 3.75e-4 --num_epochs 150 --lr_step 130 --gpus 0,1 --load_model /home/cx/Centernet/models/model_best.pth

python test.py ctdet --exp_id res50_dcn_3x_usnavy512 --dataset coco_hp --arch resdcn_50 --resume 
