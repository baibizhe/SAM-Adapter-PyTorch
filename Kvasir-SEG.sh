freqnums=0.25

#
##Kvasir-SEG 10%
#for lr  in  0.0005
#do
#  for fold  in 1
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-gt-infor.yaml\
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/labels \
#    --epoch_max 100 ;
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
##Kvasir-SEG 20%
#for lr  in  0.0005
#do
#  for fold  in  1 2 3 4 5
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-seg.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/labels \
#    --epoch_max 100 ;
#  done
#done
#Kvasir-SEG 20% ada
for lr  in  0.0005
do
  for fold  in  1
  do
    echo lr $lr   $fold
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-samADA-vit-b-kvasir-seg.yaml \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P20/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold${fold}/P80/labels \
    --epoch_max 200 ;
  done
done

#Kvasir-SEG 10% ada
for lr  in  0.0005
do
  for fold  in  1
  do
    echo lr $lr   $fold
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-samADA-vit-b-kvasir-seg.yaml \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P10/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold${fold}/P90/labels \
    --epoch_max 300 ;
  done
done
