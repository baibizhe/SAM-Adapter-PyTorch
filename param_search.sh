##Kvasir-SEG 10%  already in table
#for lr  in 0.001  0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P90/labels \
#    --epoch_max 100 ;
#  done
#done
#
#/home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#/home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#--config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#--lr 0.0001 \
#--freqnums 0.25 \
#--train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P20/imgs \
#--train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P20/labels \
#--val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P80/imgs \
#--val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P80/labels \
#--epoch_max 100 ;
#exit
#

##Kvasir-SEG 10%,
#for lr  in 0.002  0.0007
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P90/labels \
#    --epoch_max 100 ;
#  done
#done


#Kvasir-SEG 10%
for lr  in  0.0005
do
  for freqnums  in 0.25
  do
    echo lr $lr  freqnums $freqnums
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
    --lr $lr \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P10/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P10/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P90/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split1090/fold1/P90/labels \
    --epoch_max 100 ;
  done
done


#Kvasir-SEG 20%
for lr  in  0.0005
do
  for freqnums  in 0.25
  do
    echo lr $lr  freqnums $freqnums
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P20/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P20/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P80/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P80/labels \
    --epoch_max 100 ;
  done
done



#exit
#
#
##Kvasir-SEG 80%
#for lr  in 0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P80/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P80/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P20/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/Kvasir-SEG/split2080/fold1/P20/labels \
#    --epoch_max 100 ;
#  done
#done
#
#
##ETIS-LaribPolypDB 10%
#for lr  in  0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split1090/fold1/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split1090/fold1/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split1090/fold1/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split1090/fold1/P90/labels \
#    --epoch_max 100 ;
#  done
#done
#
#
##ETIS-LaribPolypDB 20%
#for lr  in 0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P20/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P20/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P80/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P80/labels \
#    --epoch_max 100 ;
#  done
#done
#
#
##ETIS-LaribPolypDB 80%
#for lr  in  0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P80/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P80/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P20/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/ETIS-LaribPolypDB/split2080/fold1/P20/labels \
#    --epoch_max 100 ;
#  done
#done
#
#
#
#
#
#
##busi 10%
#for lr  in 0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split1090/fold1/P10/imgs  \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split1090/fold1/P10/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split1090/fold1/P90/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split1090/fold1/P90/labels \
#    --epoch_max 100 ;
#  done
#done
#
#
#
#
##busi 20%
#for lr  in 0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P20/imgs  \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P20/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P80/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P80/labels \
#    --epoch_max 100 ;
#  done
#done
#
#
##busi 80%
#for lr  in 0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P80/imgs  \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P80/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P20/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/busi/split2080/fold1/P20/labels \
#    --epoch_max 100 ;
#  done
#done
#
#




#kvasir-instrument10%
for lr  in  0.0005
do
  for freqnums  in 0.25
  do
    echo lr $lr  freqnums $freqnums
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold1/P10/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold1/P10/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold1/P90/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split1090/fold1/P90/labels \
    --epoch_max 100 ;
  done
done


#kvasir-instrument 20%
for lr  in 0.0005
do
  for freqnums  in 0.25
  do
    echo lr $lr  freqnums $freqnums
    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
    --lr $lr \
    --freqnums $freqnums \
    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P20/imgs \
    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P20/labels \
    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P80/imgs \
    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P80/labels \
    --epoch_max 100 ;
  done
done

#
##kvasir-instrument 80%
#for lr  in  0.0005
#do
#  for freqnums  in 0.25
#  do
#    echo lr $lr  freqnums $freqnums
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-decoder.yaml \
#    --lr $lr \
#    --freqnums $freqnums \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P80/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P80/labels \
#    --val_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P20/imgs \
#    --val_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/kvasir-instrument/split2080/fold1/P20/labels \
#    --epoch_max 100 ;
#  done
#done



