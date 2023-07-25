Use following flags:
--load_model: for evaluation or for resuming the training
--sub_data_dir: base folder is data/kitti/training, in default setting this folder has image_2 and label_2, whereas for navigating between multiple datasets a further sub data directory is created here with the name of dataset. In code, the base folder points to "data/kitti/": hence sub_data_dir="training/<name_of_Dataset>"
--exp_id: it will creat folder with the name of exp_id in "SqueezeDet-PyTorch/exp" which will contain logs, debug files, models etc.
--image_extension_jpg: ususally images in dataset are of png and jpg. default is set for png if the dataset contains jpg pass this flag
--oneimage: will read file IDs from data/kitti/image_sets/train_oneimage.txt or data/kitti/image_sets/val_oneimage.txt instead of data/kitti/image_sets/train.txt or data/kitti/image_sets/val.txt which is default. (Beware: the "oneimage" is misleading). train or val is decided on the bases of mode
--mode: set it to "eval" or "train". trian is default. verify it in config.py file

Train Validation Split:
- The file ids are fetched from image_2 or label_2 and stored in trainval.txt
- The routine is called random_split_train_val.py which will create train.txt and val.txt with default 90/10 split. can be modified in the file

Random Exec Commands:
python main.py --exp_id="eval_synth_2.0_on_Train_Data_at_1050_epoch" --sub_data_dir="training/synth_2.0" --load_model="/workspace/SqueezeDet-PyTorch_simple_bypass/exp/train_synth_2.0/model_1050.pth" --image_extension_jpg --mode=eval --oneimage
