#! /bin/bash

export SCRIPT=$0
export TASK=$1
shift

CWD=$(pwd)

NOW=$(date +%Y%m%d_%H%M%S)

export PS4='+{$LINENO:${FUNCNAME[0]}} '

export CUDA_VISIBLE_DEVICES=$(python -c "import setGPU" | cut -d " " -f 5) || true

source activate mimic

old="$IFS"
IFS='_'
export ARGS="$*"
export ARGS="${ARGS//-/}"
IFS=$old

export PREFIX=${NOW}_${TASK}
export LOGFILE=logs/${NOW}_${TASK}_${ARGS}.log

mkdir -p logs
exec &> >(tee "$LOGFILE")
exec 2>&1

echo ">> CMD $NOW $TASK $ARGS"
echo ">> ARG $*"
echo ">> ENV GPU $CUDA_VISIBLE_DEVICES LOG $LOGFILE"
echo ">> tensorboard --logdir=$PREFIX" 

# cd  ../mimic3-benchmarks
export PYTHONPATH=$PYTHONPATH:`pwd`

# bash $CWD/$SCRIPT 2>&1 | tee LOGFILE
set -xeuo pipefail

case $TASK in 
help)
    set +x
    echo $SCRIPT download_data
    echo $SCRIPT download_demo
    echo $SCRIPT prepare
    echo $SCRIPT ihm_lr --l2 --C 0.001
    echo $SCRIPT ihm_dl --mode train  --dim 256 --depth 1 --dropout 0 
    echo $SCRIPT ihm_dl_fast --mode train  --dim 256 --depth 1 --dropout 0
    echo $SCRIPT multitask --mode train  --dim 1024 --ihm_C 0.02 --decomp_C 0.1 --los_C 0.5 --pheno_C 1.0
    echo $SCRIPT multitask_fast --mode train  --dim 1024 --ihm_C 0.02 --decomp_C 0.1 --los_C 0.5 --pheno_C 1.0
;;

default)
    echo "In-hospital mortality prediction"
    #Run the following command to train the neural network which gives the best result. We got the best performance on validation set after 28 epochs.
    cd mimic3models/in_hospital_mortality/
    python -u main.py --network ../common_keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8

    #Use the following command to train logistic regression. The best model we got used L2 regularization with C=0.001:
    cd mimic3models/in_hospital_mortality/logistic/
    python -u main.py --l2 --C 0.001

    echo "Decompensation prediction"
    # The best model we got for this task was trained for 36 chunks (that's less than one epoch; it overfits before reaching one epoch because there are many training samples for the same patient with different lengths).
    cd mimic3models/decompensation/
    python -u main.py --network ../common_keras_models/lstm.py --dim 128 --timestep 1.0 --depth 1 --mode train --batch_size 8

    # Use the following command to train a logistic regression. It will do a grid search over a small space of hyperparameters and will report the scores for every case.
    cd mimic3models/decompensation/logistic/
    python -u main.py
    
    echo "Length of stay prediction"
    # The best model we got for this task was trained for 19 chunks.
    cd mimic3models/length_of_stay/
    python -u main.py --network ../common_keras_models/lstm.py --dim 64 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8 --partition custom
    
    #Use the following command to train a logistic regression. It will do a grid search over a small space of hyperparameters and will report the scores for every case.
    cd mimic3models/length_of_stay/logistic/
    python -u main_cf.py

    echo "Phenotype classification"
    # The best model we got for this task w
    as trained for 20 epochs.
    cd mimic3models/phenotyping/
    python -u main.py --network ../common_keras_models/lstm.py --dim 256 --timestep 1.0 --depth 1 --dropout 0.3 --mode train --batch_size 8
    
    #Use the following command for logistic regression. It will do a grid search over a small space of hyperparameters and will report the scores for every case.
    cd mimic3models/phenotyping/logistic/
    python -u main.py

    echo "Multitask learning"
    #ihm_C, decomp_C, los_C and ph_C coefficients control the relative weight of the tasks in the multitask model. Default is 1.0. Multitask network architectures are stored in mimic3models/multitask/keras_models. Here is a sample command for running a multitask model.
    cd mimic3models/multitask/
    python -u main.py --network keras_models/lstm.py --dim 512 --timestep 1 --mode train --batch_size 16 --dropout 0.3 --ihm_C 0.2 --decomp_C 1.0 --los_C 1.5 --pheno_C 1.0

;;

test)
    ./run.sh test_ihm_lr
    ./run.sh test_ihm_dl
;;

ihm_lr)
    cd mimic3models/in_hospital_mortality/logistic/
    python -u main.py $@
;;

test_ihm_lr)
# usage: main.py [-h] [--C C] [--l1] [--l2]
#                [--period {first4days,first8days,last12hours,first25percent,first50percent,all}]
#                [--features {all,len,all_but_len}]
#                [--method {gridsearch,lgbm,logistic}]

    ./run.sh ihm_lr --method logistic --l1
    ./run.sh ihm_lr --method logistic --l2
    ./run.sh ihm_lr --method gridsearch
    ./run.sh ihm_lr --method lgbm

    ./run.sh ihm_lr --method gridsearch --period first4days --features all
    ./run.sh ihm_lr --method gridsearch --period first8days --features all
    ./run.sh ihm_lr --method gridsearch --period last12hours --features all
    ./run.sh ihm_lr --method gridsearch --period first25percent --features all
    ./run.sh ihm_lr --method gridsearch --period first50percent --features all
    ./run.sh ihm_lr --method gridsearch --period all --features all

;;

ihm_dl)
    cd mimic3models/in_hospital_mortality/
    python -u main.py --prefix $PREFIX --network ../common_keras_models/lstm.py $@
;;

test_ihm_dl)
    ./run.sh ihm_dl --dim 8 --depth 1 --dropout 0 
    ./run.sh ihm_dl --dim 16 --depth 1 --dropout 0 
    ./run.sh ihm_dl --dim 32 --depth 1 --dropout 0 
    ./run.sh ihm_dl --dim 64 --depth 1 --dropout 0 
    ./run.sh ihm_dl --dim 128 --depth 1 --dropout 0 
    ./run.sh ihm_dl --dim 256 --depth 1 --dropout 0 

    ./run.sh ihm_dl --dim 256 --depth 1 --dropout 0.5 
    ./run.sh ihm_dl --dim 256 --depth 1 --dropout 0.7 
    ./run.sh ihm_dl --dim 256 --depth 1 --dropout 0.9 
;;

ihm_dl_fast)
    cd mimic3models/in_hospital_mortality/
    python -u main.py --prefix $PREFIX --network ../common_keras_models/lstm_fast.py $@
    cp keras_logs/$PREFIX.* ../../
;;

multitask)
    cd mimic3models/multitask/
    python -u main.py --network keras_models/lstm.py  $@
;;

multitask_fast)
    cd mimic3models/multitask/
    python -u main.py --network keras_models/lstm.py $@
;;

prepare)
    MIMIC3DATA=$1
    # if [ -e data ]
    # then
    #     echo "data exist, move to data_$NOW"
    #     mv data data_$NOW
    # fi
    python scripts/extract_subjects.py $MIMIC3DATA data/root/
    #python scripts/extract_subjects.py ../mimic3demo data/root/

    python scripts/validate_events.py data/root/
    python scripts/extract_episodes_from_subjects.py data/root/
    python scripts/split_train_and_test.py data/root/
    python scripts/create_in_hospital_mortality.py data/root/ data/in-hospital-mortality/
    python scripts/create_decompensation.py data/root/ data/decompensation/
    python scripts/create_length_of_stay.py data/root/ data/length-of-stay/
    python scripts/create_phenotyping.py data/root/ data/phenotyping/
    python scripts/create_multitask.py data/root/ data/multitask/

    python mimic3models/split_train_val.py in-hospital-mortality
    python mimic3models/split_train_val.py decompensation
    python mimic3models/split_train_val.py length-of-stay
    python mimic3models/split_train_val.py phenotyping
    python mimic3models/split_train_val.py multitask
;;

download_data)
# Full dataset
    mkdir mimic3data_$NOW
    cd mimic3data_$NOW
    wget --user 382641632@qq.com --password cony1986ni -A csv.gz -m -p -E -k -K -np -nd https://physionet.org/works/MIMICIIIClinicalDatabase/files/
    echo "download dataset in mimic3data_$NOW"
;;

download_demo)
# Demo dataset
# The demo dataset contains data for 100 patients and excludes the noteevents table.
    mkdir mimic3demo_$NOW
    cd mimic3demo_$NOW
    wget --user 382641632@qq.com --password cony1986ni  -A csv.gz -m -p -E -k -K -np https://physionet.org/works/MIMICIIIClinicalDatabaseDemo/files/
    echo "download dataset in mimic3demo_$NOW"
;;

verify_data)
    md5sum *.gz > md5sum2.txt
    cat md5sum2.txt
;;
esac

set +x
NOW=$(date +%Y%m%d_%H%M%S)
echo ">> END $NOW $TASK $ARGS"
