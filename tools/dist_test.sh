#!/usr/bin/env bash

#CONFIG=$1
#CHECKPOINT=$2
#GPUS=$3
#NNODES=${NNODES:-1}
#NODE_RANK=${NODE_RANK:-0}
#PORT=${PORT:-29501}
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
#
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch \
#    --nnodes=$NNODES \
#    --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \
#    --master_port=$PORT \
#    $(dirname "$0")/test.py \
#    $CONFIG \
#    $CHECKPOINT \
#    --launcher pytorch \
#    ${@:4}

#shell命令中参数传递，$0固定，代表执行的文件名，$n代表传入的第n个参数
#配置文件路径
CONFIG=$1
#模型文件路径
CHECKPOINT=$2
GPUS=$3
#设置变量的默认值
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

#添加环境变量(FlashOCC-master)，和sys.path.append命令作用相同
#将配置文件的上级目录添加到Python解释器搜索模块的路径中
#dirname获取文件所在的目录
#nnodes，节点的数量，通常一个节点对应一个主机，node_rank，节点的序号，从0开始
#master_addr，0号主机的IP地址，nproc_per_node，一个节点中显卡的数量
#${@:4}，从第四个参数开始，获取所有的位置参数
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
