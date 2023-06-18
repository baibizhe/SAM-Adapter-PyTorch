freqnums=0.25
#pra-net   50%
for freqnums  in    0.25
  do
  for lr  in    0.0005
  do
    for fold  in  1
    do
      echo lr $lr   $fold $freqnums
      /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
      /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
      --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-l-pra-net-data.yaml \
      --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold$fold/P51/imgs \
      --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split4951/fold$fold/P51/labels \
      --lr $lr \
      --freqnums $freqnums \
      --epoch_max 80 ;
    done
  done
done
#done


#pra-net 10%
#for lr  in  0.0007
#do
#  for fold  in  2
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-ADAvit-b-pra-net-data.yaml \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split1090/fold$fold/P10/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split1090/fold$fold/P10/labels \
#    --lr $lr \
#    --freqnums $freqnums \
#    --epoch_max 200 ;
#  done
#done




# pra-net 20%
#for lr  in  0.0005
#do
#  for fold  in 1
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-ADAvit-b-pra-net-data.yaml \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P20/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P20/labels \
#    --lr $lr \
#    --freqnums $freqnums \
#    --epoch_max 200 ;
#  done
#done


#pra-net   50%
#for freqnums  in    0.25 0.005 1
#  do
#  for lr  in    0.0005
#  do
#    for fold  in  1  #6.4 fold 5 0.7711
#    do
#      echo lr $lr   $fold $freqnums
#      /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#      /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#      --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-pra-net-data.yaml \
#      --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P80/imgs \
#      --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P80/labels \
#      --lr $lr \
#      --freqnums $freqnums \
#      --epoch_max 100 ;
#    done
#  done
#done







##pra-net  ADA 20%
#for lr  in    0.0007
#do
#  for fold  in 1
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-ADAvit-b-pra-net-data.yaml \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P80/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P80/labels \
#    --lr $lr \
#    --freqnums $freqnums \
#    --epoch_max 80 ;
#  done
#done

## pra-net 80%
#for lr  in  0.0005
#do
#  for fold  in 1
#  do
#    echo lr $lr   $fold
#    /home/ubuntu/anaconda3/envs/samEnv/bin/python /home/ubuntu/anaconda3/envs/samEnv/lib/python3.8/site-packages/torch/distributed/launch.py --nnodes 1 --nproc_per_node 1 \
#    /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/train.py  \
#    --config  /home/ubuntu/works/code/working_proj/SAM-Adapter-PyTorch/configs/cod-sam-vit-b-pra-net-data.yaml \
#    --train_img_dir /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P80/imgs \
#    --train_label_dir  /home/ubuntu/works/code/working_proj/segment-anything/data/pra_net_dataset/TrainDataset/split2080/fold$fold/P80/labels \
#    --lr $lr \
#    --freqnums $freqnums \
#    --epoch_max 50 ;
#  done
#done
#
