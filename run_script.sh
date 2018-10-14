#!/usr/bin/env bash

python3 script/train.py --lr 0.0001 --dropout 0.70 --model_name _0001_70
python3 script/train.py --lr 0.00001 --dropout 0.70 --model_name _00001_70
python3 script/train.py --lr 0.001 --dropout 0.70 --model_name _001_70
python3 script/train.py --lr 0.0001 --dropout 0.55 --model_name _0001_55
python3 script/train.py --lr 0.00001 --dropout 0.55 --model_name _00001_55
python3 script/train.py --lr 0.001 --dropout 0.55 --model_name _001_55
python3 script/train.py --lr 0.0001 --dropout 0.40 --model_name _0001_40
python3 script/train.py --lr 0.00001 --dropout 0.40 --model_name _00001_40
python3 script/train.py --lr 0.001 --dropout 0.40 --model_name _001_40
