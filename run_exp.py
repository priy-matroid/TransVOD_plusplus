import os
################# vid single ####################


command_list = [
    "python", "./main.py",
    # "--pretrainedmodel", "./pretrained_checkpoints/checkpoint0005.pth",            
    "--backbone", "swin_b_p4w7",
    "--epochs", "5",
    "--eval",
    "--num_feature_levels", "1", 
    "--num_queries", "100", 
    "--hidden_dim", "256",
    "--dilation",
    "--batch_size", "1", 
    "--num_ref_frames", "2", 
    "--num_classes", "4",
    "--img_side", "600", 
    "--resume" , "./experiments/UAV_my_sample/checkpoint0007.pth",
    "--lr_drop_epochs", "4",
    "--num_workers", "24",
    "--with_box_refine",
    # "--coco_pretrain",
    "--dataset_file", "vid_multi",
    "--output_dir", "./experiments/UAV_high_res_test/" ,
    "--data_root", "/home/ubuntu/priy_dev/Datasets/UAV/",
    # "--freeze_backbone",
            ]

# command_list = [
#     "python", "./main.py",
#     # "--pretrainedmodel", "./pretrained_checkpoints/checkpoint0005.pth",            
#     "--backbone", "swin_b_p4w7",
#     # "--eval", 
#     "--epochs", "50",
#     "--num_feature_levels", "1", 
#     "--num_queries", "100", 
#     "--hidden_dim", "256",
#     "--dilation",
#     "--batch_size", "3", 
#     # "--num_ref_frames", "14", 
#     "--num_classes", "4",
#     "--img_side", "1080", 
#     # "--resume" , "./pretrained_checkpoints/coco_backbone/swinb_checkpoint0048.pth",
#     "--resume" , "./pretrained_checkpoints/single/transvod_pp_single_swinb_checkpoint0006.pth",
#     "--lr_drop_epochs", "30","45",
#     "--num_workers", "24",
#     "--with_box_refine",
#     "--coco_pretrain",
#     # "--eval",
#     "--dataset_file", "vid_single",
#     "--output_dir", "./experiments/UAV_higher_res_single/" ,
#     "--data_root", "/home/ubuntu/priy_dev/Datasets/UAV/",
#     # "--freeze_backbone",
#             ]
command = ' '.join(command_list)

os.system(command)


