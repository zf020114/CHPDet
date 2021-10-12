cd src
# train
python main.py multi_pose --exp_id hg_3x --dataset coco_hp --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 -load_model ../models/ctdet_coco_hg.pth --gpus 0,1,2,3,4 --num_epochs 150 --lr_step 130
# or use the following command if your have dla_1x trained
# python main.py multi_pose --exp_id hg_3x --dataset coco_hp  --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 --gpus 0,1,2,3,4 --num_epochs 150 --lr_step 130 --load_model ../exp/multi_pose/hg_1x/model_40.pth --resume
# test
python test.py multi_pose --exp_id hg_3x --dataset coco_hp --arch hourglass --keep_res --resume
# flip test
python test.py multi_pose --exp_id hg_3x --dataset coco_hp --arch hourglass --keep_res --resume --flip_test
cd ..

python main.py multi_pose --exp_id hg_3x_DOTA800512 --dataset coco_hp --arch hourglass --batch_size 10 --master_batch 5 --lr 2.5e-4 --load_model /home/cx/Centernet_tz/exp/multi_pose/hg_3x_DOTA800512/model_last.pth --gpus 0,1 --num_epochs 150 --lr_step 130

python main.py multi_pose --exp_id hg_ori_3x_DOTA800512 --dataset coco_hp --arch hourglass --batch_size 10 --master_batch 5 --lr 2.5e-4 --load_model /home/cx/Centernet_tz/exp/multi_pose/hg_3x_DOTA800512/model_last.pth --gpus 0,1 --num_epochs 150 --lr_step 130

python main.py multi_pose --exp_id hg_3x_DOTA8001024 --dataset coco_hp --arch hourglass --batch_size 2 --master_batch 1 --lr 2.5e-4 --load_model /home/cx/Centernet_tz/exp/multi_pose/hg_3x_DOTA800512/model_last.pth --gpus 0,1 --num_epochs 150 --lr_step 130
