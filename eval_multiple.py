import os


for each_model in range(229999, 270000, 5000):
    each_model = 'model_0' + str(each_model) + '.pth'
    
    print(each_model)
    cmd = "python tools/train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/my_mask_rcnn_ResNeSt_200_FPN_syncBN_all_tricks_3x_move_lr.yaml --eval-only MODEL.WEIGHTS /home/ubuntu/detectron2-ResNeSt/output_200_all_tricks/" + each_model

    os.system(cmd)
