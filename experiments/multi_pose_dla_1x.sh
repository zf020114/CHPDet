cd src
# train
python main.py multi_pose --exp_id dla_1x --dataset coco_hp --batch_size 128 --master_batch 9 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0,1,2,3,4,5,6,7 --num_workers 16
# test
python test.py multi_pose --exp_id dla_1x --dataset coco_hp --keep_res --resume
# flip test
python test.py multi_pose --exp_id dla_1x --dataset coco_hp --keep_res --resume --flip_test
cd ..

python main.py multi_pose --exp_id dla_1x --dataset coco_hp --batch_size 16 --master_batch 16 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 1--num_workers 1
python main.py ctdet --exp_id coco_dla_1x --batch_size 128 --master_batch 9 --lr 5e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16

multi_pose --exp_id dla_1x --dataset coco_hp --batch_size 32 --master_batch 16 --lr 1.25e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0,1 --num_workers 2
python main.py multi_pose --exp_id dla_1x --dataset coco_hp --batch_size 24 --master_batch 12 --lr 1.25e-4 --load_model /home/cx/Centernet/models/ctdet_coco_dla_2x.pth --gpus 0,1 --num_workers 8
