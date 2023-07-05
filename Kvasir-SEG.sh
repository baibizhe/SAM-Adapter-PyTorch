freqnums=0.25
#Kvasir-SEG 5%
#
#for lr  in  0.0005
#do
#  for fold  in 1 3 5
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-l-kvasir-seg.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split5-95/fold${fold}/P5/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split5-95/fold${fold}/P5/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split5-95/fold${fold}/P95/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split5-95/fold${fold}/P95/labels \
#    --epoch_max 150 ;
#  #    --val_img_h 768 \
#  done
#done
#    --val_img_w 768 \




#
#Kvasir-SEG 10%
for lr  in  0.0005
do
  for fold  in 1
  do
    echo lr $lr   $fold
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-seg.yaml \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/labels \
    --epoch_max 80 ;
#        --val_img_w 352 \
#    --val_img_h 352 \
  done
done



#
#
##Kvasir-SEG 20%
#for lr  in  0.0005
#do
#  for fold  in 1 3
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-l-kvasir-seg.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/labels \
#    --epoch_max 80 ;
#
##    --val_img_w 768 \
##    --val_img_h 768 \
#  done
#done
#
#





#Kvasir-SEG 10%
#for lr  in  0.0005
#do
#  for fold  in 1 3
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-l-kvasir-seg.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/labels \
#        --val_img_w 352 \
#    --val_img_h 352 \
#    --epoch_max 80 ;
#  done
#done


##Kvasir-SEG 50%
#for lr  in  0.0005
#do
#  for fold  in 1 3
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-l-kvasir-seg.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split4060/fold${fold}/P40/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split4060/fold${fold}/P40/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split4060/fold${fold}/P60/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split4060/fold${fold}/P60/labels \
#        --val_img_w 352 \
#    --val_img_h 352 \
#    --epoch_max 40 ;
#  done
#done
##Kvasir-SEG 10%
#for lr  in  0.0005
#do
#  for fold  in 1 2 3 4 5
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-seg.yaml\
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/labels \
#    --epoch_max 100 ;
#  done
#done
#
#

##Kvasir-SEG 20% ada
#for lr  in  0.0005
#do
#  for fold  in  1
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-samADA-vit-b-kvasir-seg.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/labels \
#    --epoch_max 200 ;
#  done
#done
#
##Kvasir-SEG 10% ada
#for lr  in  0.0005
#do
#  for fold  in  1
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-samADA-vit-b-kvasir-seg.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/labels \
#    --epoch_max 300 ;
#  done
#done
