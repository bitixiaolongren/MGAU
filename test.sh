#!/bin/bash
date


python train.py --bata53 0.3 --bata43 0.7 --bata52 0.2 --bata42 0.3 --bata32 0.5 --in_size_h 321 --in_size_w 321 --test_size_h 321 --test_size_w 321 --batch_size 8 --epochs 100   
#python train.py --bata53 0.3 --bata43 0.7 --bata52 0.2 --bata42 0.3 --bata32 0.5 --in_size_h 512 --in_size_w 512 --test_size_h 321 --test_size_w 321 --batch_size 8 --epochs 100     


