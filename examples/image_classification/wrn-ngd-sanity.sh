echo 'Running'

. ./path.sh

STATE_DICT="wrn-reparametrized.pth"
EXP_DIR="vs-smalllr-largebs"

python ./cifar_reparam.py \
    --exp $EXP_DIR/SGD/ \
    -a wrn_reparametrized \
    --depth 28 \
    --widen-factor 10 \
    --train-batch 256
    --optimizer sgd \
    --epochs 50 \
    --scheduler step \
    --milestones 1000 \
    --gamma 0.1 \
    --wd 0.0 \
    --ngd-alpha 1 \
    --update-period 1 \
    --state-dict $STATE_DICT \
    

python ./cifar_reparam.py \
    --exp $EXP_DIR/NGD/ \
    -a wrn_reparametrized \
    --depth 28 \
    --widen-factor 10 \
    --train-batch 256
    --optimizer ngd \
    --epochs 50 \
    --scheduler step \
    --milestones 1000 \
    --gamma 0.1 \
    --wd 0.0 \
    --ngd-alpha 1 \
    --update-period 1 \
    --state-dict $STATE_DICT \


python ./cifar_reparam.py \
    --exp $EXP_DIR/SGD-reparametrized/ \
    -a wrn_reparametrized \
    --depth 28 \
    --widen-factor 10 \
    --train-batch 256
    --optimizer sgd \
    --epochs 50 \
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
    -a wrn_reparametrized \
    --depth 28 \
    --widen-factor 10 \
    --train-batch 256
    --optimizer ngd \
    --epochs 50 \
    --scheduler step \
    --milestones 1000 \
    --gamma 0.1 \
    --wd 0.0 \
    --ngd-alpha 1 \
    --update-period 1 \
    --reparametrized \
    --state-dict $STATE_DICT \

