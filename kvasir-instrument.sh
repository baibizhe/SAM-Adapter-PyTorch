freqnums=0.25


#done
#kvasir-instrument10%
#for lr  in  0.0005
#do
#  for fold  in 1
#  do
#    echo lr $lr  freqnums fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-instrument.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P90/labels \
#    --val_img_w  512 \
#    --val_img_h  288 \
#    --epoch_max 100 ;
#  done
#done

#kvasir-instrument20%
#for lr  in  0.0005
#do
#  for fold  in 1 2 3 4 5
#  do
#    echo lr $lr  freqnums fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-instrument.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P80/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P80/labels \
#    --val_img_w  512 \
#    --val_img_h  288 \
#    --epoch_max 100 ;
#  done
#done

#kvasir-instrument2%  val on 20%
for lr  in  0.0002
do
  for fold  in  3 4 5
  do
    echo lr $lr  freqnums fold
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-l-kvasir-instrument.yaml \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split298/fold${fold}/P2/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split298/fold${fold}/P2/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/labels \
    --val_img_w  1024 \
    --val_img_h  1024 \
    --epoch_max 80 ;
  done
done
#
##kvasir-instrument10% val on 20%
#for lr  in  0.0005
#do
#  for fold  in 1 2 3
#  do
#    echo lr $lr  freqnums fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-instrument.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/labels \
#    --val_img_w  1024 \
#    --val_img_h  1024 \
#    --epoch_max 100 ;
#  done
#done

##kvasir-instrument10%
#for lr  in  0.0005
#do
#  for fold  in 5 4 3 2 1
#  do
#    echo lr $lr  freqnums fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-instrument.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold${fold}/P90/labels \
#    --val_img_w  512 \
#    --val_img_h  288 \
#    --epoch_max 100 ;
#  done
#done


#kvasir-instrument 20%
#for lr  in  0.0005
#do
#  for fold  in 5 4 3 2 1
#  do
#    echo lr $lr  freqnums fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-kvasir-instrument.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P20/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P80/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold${fold}/P80/labels \
#    --val_img_w  512 \
#    --val_img_h  288 \
#    --epoch_max 100 ;
#  done
#done

