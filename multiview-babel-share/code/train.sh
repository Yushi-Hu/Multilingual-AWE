#!/bin/bash

#SBATCH -x gpu-g3,gpu-g38
#SBATCH --partition=speech-gpu
#SBATCH --output=train_log.txt
#SBATCH --open-mode=append

set -e

src="$1"
config="$2"
data_dir="$3"
cache_dir="$4"

export PYTHONPATH="$src"

echo "Hostname: $(hostname)"
echo "PYTHONPATH=$PYTHONPATH"
echo "Config: $config"

if [ -z $SLURM_JOB_ID ]; then
  python $src/train.py --config="$config"
else
  echo "updating cache directory..."
  start_t=`date +%s`
  if [ ! -d $cache_dir ]
      then
      echo "creating cache directory..."
      mkdir -p $cache_dir
  fi
  rsync --update -raz --progress $data_dir/* $cache_dir
  end_t=`date +%s`
  left_s=$(expr $(expr 230 \* 60) - $((end_t-start_t)))
  timeout --foreground --signal=SIGUSR1 ${left_s}s \
    python $src/train.py --config="$config"
fi
