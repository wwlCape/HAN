## train
# BI, scale 2, 3, 4, 8
##################################################################################################################################
# BI, scale 2, 3, 4, 8
# RCAN_BIX2_G10R20P48, input=48x48, output=96x96
#LOG=./../experiment/RCAN_BIX2_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96 2>&1 | tee $LOG

# RCAN_BIX3_G10R20P48, input=48x48, output=144x144
LOG=./../experiment/RCAN_BIX3_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN1 --save RCAN_BIX3_G10R20P48 --scale 3 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG

# RCAN_BIX4_G10R20P48, input=48x48, output=192x192

#LOG=./../experiment/MatrixModel_BIX4_G1R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0,1 python main.py --model MatrixModel --save MatrixModel_BIX4_G1R20P48 --scale 4 --n_resgroups 1 --n_resblocks 20 --reset --chop --save_results --print_model --patch_size 192 --pre_train /media/zrh/cc9cb710-2fc7-4382-81ff-649502a83b92/RCAN-master/RCAN_TrainCode/experiment/MatrixModel_BIX4_G1R20P48/model/model_best.pt 2>&1 | tee $LOG

# RCAN_BIX8_G10R20P48, input=48x48, output=384x384
#LOG=./../experiment/RCAN_BIX8_G10R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save RCAN_BIX8_G10R20P48 --scale 8 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 384 --pre_train ../experiment/model/RCAN_BIX2.pt 2>&1 | tee $LOG

#LOG=./../experiment/MatrixModelD_BIX4_G1R20P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model MatrixModel --save MatrixModelD_BIX4_G1R20P48 --scale 4 --n_resgroups 1 --n_resblocks 20 --reset --chop --save_results --print_model --patch_size 192  2>&1 | tee $LOG
