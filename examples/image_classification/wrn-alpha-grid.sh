echo 'Running'

. ./path.sh
for SEED in 42 69 101
do 
    for i in 0.01 1.0 2.0 4.0 6.0 9.0
    do
        python ./cifar.py \
            --exp exp/cifar/alpha-grid/a$i/wrn-28-10-ngd/seed$SEED \
            -a wrn \
            --depth 28 \
            --widen-factor 10 \
            --optimizer ngd \
            --epochs 50 \
            --scheduler step \
            --milestones 38 \
            --gamma 0.1 \
            --wd 1e-4 \
            --ngd-alpha $i \
            --manualSeed $SEED 
    done
done