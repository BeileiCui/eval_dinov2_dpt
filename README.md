# eval_dinov2_dpt
A repository to evaluate the dinov2 and dpt model on the SCARED dataset

steps:
1. Follow the instructions in [dinov2](https://github.com/facebookresearch/dinov2) and [dpt](https://github.com/isl-org/DPT) to prepare for the environment and dependencies.
2. Change the ```sys.path.insert``` path to your own path in the ```run_monodepth_scared.py``` files in each folder. Also, change the arguments in the files to your own path. Make sure your path to split file is correct.
3. run  ```dinov2/run_monodepth_scared.py``` and ```dpt/run_monodepth_scared.py``` to start the evaluation.
