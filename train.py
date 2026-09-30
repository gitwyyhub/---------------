import torch
import torch.nn as nn
import torch.optim as optim
from option import get_args
from getdata import MyData
from model import My_CNN, ResNet
from utils import make_dir, draw_number
import numpy as np


opt = get_args()


def train():
    args = opt

    # ---- 1. 准备设备、数据、模型 ----
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    make_dir()

    dataloaders = MyData()
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    # 使用 ImageNet 预训练的 ResNet50 做迁移学习，替换分类头为 3 类输出
    model = ResNet(num_classes=3, pretrained=True).to(device)

    # 断点续训：加载之前保存的模型权重继续训练
    if args.pretrained:
        print('Loading checkpoint: {}'.format(args.which_epoch))
        ckpt = torch.load(args.checkpoints + args.which_epoch, map_location=device)
        model.load_state_dict(ckpt, strict=False)

    # ---- 2. 损失函数、优化器、学习率调度 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # StepLR: 每 step_size=10 个 epoch，学习率衰减为原来的 gamma=0.1 倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = args.epochs
    best_acc = 0.0

    # 记录每个 epoch 的损失和准确率，用于训练结束后绘图
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    # 初始化日志文件，写入 CSV 表头
    with open(args.logging_txt, 'a') as f:
        f.write('Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n')

    # ---- 3. 训练循环 ----
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 40)

        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()          # 清空上一轮的梯度
            outputs = model(images)         # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()                 # 反向传播，计算梯度
            optimizer.step()               # 更新模型参数

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 取 logits 最大值作为预测类别
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_loss_list.append(train_loss_avg)
        train_acc_list.append(train_acc)

        # ========== 验证阶段 ==========
        model.eval()  # 切换到评估模式（关闭 Dropout 和 BN 的训练行为）
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # 关闭自动求导，节省显存并加速推理
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss_avg = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_loss_list.append(val_loss_avg)
        val_acc_list.append(val_acc)

        scheduler.step()  # 每个 epoch 结束后更新学习率

        print('Train Loss: {:.4f}  Train Acc: {:.4f}'.format(train_loss_avg, train_acc))
        print('Val Loss:   {:.4f}  Val Acc:   {:.4f}'.format(val_loss_avg, val_acc))

        # 将本轮结果写入日志文件（CSV 格式）
        with open(args.logging_txt, 'a') as f:
            f.write('{},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                epoch + 1, train_loss_avg, train_acc, val_loss_avg, val_acc))

        # 验证准确率超过历史最佳时保存模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.checkpoints + 'best.pth')
            print('Best model saved, val_acc: {:.4f}'.format(best_acc))

        # 每隔 save_freq 轮保存一次检查点
        if (epoch + 1) % args.save_freq == 0:
            torch.save(model.state_dict(), args.checkpoints + 'epoch_{}.pth'.format(epoch + 1))

    # ---- 4. 训练结束：绘制 Loss/Accuracy 曲线 ----
    epoch_list = np.arange(1, epochs + 1)
    draw_number(epoch_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list)

    print('Training complete. Best val_acc: {:.4f}'.format(best_acc))


if __name__ == '__main__':
    setattr(opt, 'save_freq', 5)
    train()