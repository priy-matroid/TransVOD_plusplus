import os
################# vid single ####################


command_list = [
    "python", "./main.py",
    "--backbone", "swin_b_p4w7",
    "--epochs", "16",
    "--eval",
    "--num_feature_levels", "1", 
    "--num_queries", "100", 
    "--hidden_dim", "256",
    "--dilation",
    "--batch_size", "1", 
    "--num_ref_frames", "14", 
    "--num_classes", "3",
    "--img_side", "600", 
    "--resume" , "./experiments/UAV_my_sample/checkpoint0007.pth",
    "--lr_drop_epochs", "9", "14"
    "--num_workers", "24",
    "--with_box_refine",
    "--dataset_file", "vid_multi",
    "--output_dir", "./experiments/UAV_high_res_test/" ,
    "--data_root", "/home/ubuntu/priy_dev/Datasets/UAV/",
            ]

command = ' '.join(command_list)

os.system(command)


