echo 'Running'

. ./path.sh

STATE_DICT="vgg-reparametrized-seed42.pth"
EXP_DIR="exp/vgg16/vs-lr1e-2-bs16384/"
BATCH_SIZE=16384
EPOCHS=1000


python ./cifar_reparam.py \
    --exp $EXP_DIR/SGD/ \
    -a vgg16 \
    --lr 0.01 \
    --train-batch $BATCH_SIZE \
    --test-batch $BATCH_SIZE \
    --optimizer sgd \
    --epochs $EPOCHS \
    --scheduler step \
    --milestones 1000 \
    --gamma 0.1 \
    --wd 0.0 \
    --ngd-alpha 1 \
    --update-period 1 \
    --state-dict $STATE_DICT \
    

python ./cifar_reparam.py \
    --exp $EXP_DIR/NGD/ \
    -a vgg16 \
    --lr 0.01 \
    --train-batch $BATCH_SIZE \
    --test-batch $BATCH_SIZE \
    --optimizer ngd \
    --epochs $EPOCHS \
    --scheduler step \
    --milestones 1000 \
    --gamma 0.1 \
    --wd 0.0 \
    --ngd-alpha 1 \
    --update-period 1 \
    --state-dict $STATE_DICT \


python ./cifar_reparam.py \
    --exp $EXP_DIR/SGD-reparametrized/ \
    -a vgg16 \
    --lr 0.01 \
    --train-batch $BATCH_SIZE \
    --test-batch $BATCH_SIZE \
    --optimizer sgd \
    --epochs $EPOCHS \
    --scheduler step \
    --milestones 1000 \
    --gamma 0.1 \
    --wd 0.0 \
    --ngd-alpha 1 \
    --update-period 1 \
    --reparametrized \
    --state-dict $STATE_DICT \
    

python ./cifar_reparam.py \
    --exp $EXP_DIR/NGD-reparametrized/ \
    -a vgg16 \
    --lr 0.01 \
    --train-batch $BATCH_SIZE \
    --test-batch $BATCH_SIZE \
    --optimizer ngd \
    --epochs $EPOCHS \
    --scheduler step \
    --milestones 1000 \
    --gamma 0.1 \
    --wd 0.0 \
    --ngd-alpha 1 \
    --update-period 1 \
    --reparametrized \
    --state-dict $STATE_DICT \

