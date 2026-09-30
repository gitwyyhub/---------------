import argparse

def get_args():
    parser = argparse.ArgumentParser(description='all argument')
    parser.add_argument('--device', type=str, default='cuda', help='可以选择cuda或者cpu训练，苹果电脑m1芯片也可以选择mps加速训练')
    parser.add_argument('--loadsize', type=int, default=224, help='统一图像尺寸')
    parser.add_argument('--epochs', type=int, default=3, help='总的训练次数')
    parser.add_argument('--batch_size', type=int, default=16, help='每次喂多少数据给到网络')
    parser.add_argument('--lr', type=float, default=1e-2, help='初始学习率')
    parser.add_argument('--dataset_train', type=str, default='./datasets/Augmented/train_classified', help='训练集路径')
    parser.add_argument('--dataset_val', type=str, default="./datasets/Augmented/val_classified", help='验证集路径')
    parser.add_argument('--dataset_test', type=str, default="./datasets/Augmented/test_classified", help='测试集路径')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型存放路径')
    parser.add_argument('--log_dir', type=str, default='./log_dir', help='训练日志保存的路径')
    parser.add_argument('--logging_txt', type=str, default='./log_dir/logging.txt', help='训练日志位置')
    parser.add_argument('--pretrained', type=bool, default=False, help='是否要继续上次的训练')
    parser.add_argument('--which_epoch', type=str, default='best.pth', help='如果继续训练，需要加载哪一个模型')
    parser.add_argument('--test_model_path', type=str, default='./checkpoints/best.pth', help='选择一个模型用于测试')
    parser.add_argument('--onnx_path', type=str, default='./checkpoints/best.onnx', help='.onnx模型的存放路径')
    parser.add_argument('--test_img_path', type=str, default='./datasets/Augmented/test_classified/WBC', help='选择一张测试图像')
    parser.add_argument('--test_dir_path', type=str, default='./datasets/Augmented/test_classified', help='选择一个测试路径')

    return parser.parse_args()