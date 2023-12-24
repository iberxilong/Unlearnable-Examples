import random
import sys
import time
import torch
import focal_loss as flc
from torch import nn, optim
from torch import optim
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0' # 下面老是报错 shape 不一致

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from Model import model as Conformer
from multi_scale_ori import MSResNet
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast


class My_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return flr_loss(3,x,y,alpha=1)

# def cer_loss(y_true,y_pred):
#
#     n = 10
#
#     def cer(y_true, y_pred):
#         index = [i for i in range(len(y_true))]
#         """ For BO, remove it for release """
#         criterion = nn.CrossEntropyLoss()
#         # ce = criterion(y_true, y_pred)
#         ce = criterion(y_pred,y_true)
#
#         ce_shuffled = 0.0
#
#         for i in range(n):
#             random.shuffle(index)
#             y_true_shuffled = y_true[index]
#             # ce_shuffled += criterion(y_true_shuffled, y_pred)
#             ce_shuffled += criterion(y_pred,y_true_shuffled)
#
#         ce_shuffled = ce_shuffled / n
#
#         return ce / ce_shuffled
#
#     return cer(y_true, y_pred)

def flr_loss(n,y_true, y_pred, alpha=None, gamma=2.0):
    fl_num = flc.categorical_focal_loss(alpha=alpha
                                    , gamma=gamma,lamda = 1.0)
    fl_de = flc.categorical_focal_loss(alpha=alpha
                                        , gamma=gamma, lamda=0.0)

    def flr(y_true, y_pred):
        index = [i for i in range(len(y_true))]
        ce = fl_num(y_true, y_pred)

        ce_shuffled = 0.0

        for i in range(n):
            random.shuffle(index)
            y_true_shuffled = y_true[index]
            ce_shuffled += fl_de(y_true_shuffled, y_pred)

        ce_shuffled = ce_shuffled / n

        return ce / ce_shuffled

    return flr(y_true, y_pred)


def findlabel(label):
    return torch.tensor((label - label % 16) / 16, dtype=torch.int64)


def train(class_num, database, epoch_size, model_save_addr, target, dim, head_num, encoder_num, learning_rate, dropout,
          traces_num, device):
    time_start = time.time()

    # 载入模型(class_num, embed_dim, head_num, encoder_num, dropout, trace_point)
    # model = Conformer(class_num, dim, head_num, encoder_num, dropout, Seq, device).to(device)
    model = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=class_num).to(device)
    model.initialize_weights()
    # model.load_state_dict(torch.load('model_param/E50B60N99000_Dim128H4Encoder2Lr0001Drop_3.pkl'))       # 继续加载模型训练
    # print(model)

    # 设置优化器和loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = My_loss()  # Loss
    # loss_function = nn.CrossEntropyLoss()  # Loss

    # 保存tensorboard的loss
    tensorboardX_writer = SummaryWriter(log_dir='logs')

    patience = 5000
    last_loss = 99.99

    best_loss = 99.999
    for epoch in range(epoch_size):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()

                train_num = 0
                train_loss = 0.0
                train_correct = 0.0

                loop = tqdm(database[phase], total=len(database[phase]), file=sys.stdout)
                for i, (inputs, labels) in enumerate(loop):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # label2 = findlabel(labels)
                    # label2 = label2.to(device)
                    optimizer.zero_grad()

                    predict = model(inputs)
                    # predict = Calpro(predict[0].cpu(), predict[2].cpu())
                    loss = loss_function(labels,predict[0])
                    # loss2 = loss_function(predict[2], label2)  # ++

                    # loss = loss1 + loss2  # ++
                    # 常规的反向传播
                    loss.backward()
                    optimizer.step()
                    # scheduler.step(finally_loss)

                    # 记录loss和命中数量
                    train_loss += loss.item()
                    train_correct += (predict[0].argmax(1) == labels).sum().item()
                    train_num += labels.shape[0]

                    # 显示进度条
                    loop.set_description(f'[{epoch + 1}/{epoch_size}]')

                # 计算结果
                finally_loss = train_loss / (i + 1)
                finally_acc = train_correct / train_num

                # tensorboardX记录loss
                write_logs('logs/train_loss.txt', finally_loss)
                write_logs('logs/train_acc.txt', finally_acc)
                tensorboardX_writer.add_scalar('train_loss', finally_loss, global_step=epoch)
                tensorboardX_writer.add_scalar('train_acc', finally_acc, global_step=epoch)

                # 显示进度条
                print('train loss:', str(finally_loss), '   train_acc:', str(finally_acc))


            else:
                model.eval()
                val_correct = 0.0
                val_num = 0.0
                val_loss = 0.0
                with torch.no_grad():
                    for i, (inputs, labels) in enumerate(database[phase]):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        # output = Calpro(outputs[0], outputs[2])
                        output = outputs[0]
                        loss = loss_function(labels,output)

                        # 记录loss和命中数量
                        val_loss += loss.item()
                        val_correct += (output.argmax(1) == labels).sum().item()

                        val_num += labels.size(0)

                    val_loss = val_loss / (i + 1)
                    val_correct = val_correct / val_num

                    # model_save_addr = 'epoch/E15B50N99000_Dim64H4E2Lr0001Drop_' +'epoc' +str(epoch) + '.pkl'
                    # torch.save(model.state_dict(), model_save_addr)

                    # 选择最好的loss进行保存
                    if val_loss <= best_loss:
                        best_loss = val_loss
                        torch.save(model.state_dict(), model_save_addr)
                        # 查看第几个epoch停止保存
                        print('保存到了第', epoch, '个epoch')
                    else:
                        torch.save(model.state_dict(), 'epoch/E15B50N99000_Dim64H4E2Lr0001Drop_' +'epoc' +str(epoch) + '.pkl')
                    # 显示进度条
                    print('val_loss:', val_loss, '   val_acc:', val_correct, '\n')


                    # tensorboardX记录loss
                    write_logs('logs/val_loss'+str(target)+'.txt', val_loss)
                    write_logs('logs/val_acc'+str(target)+'.txt', val_correct)
                    tensorboardX_writer.add_scalar('val_loss', val_loss, global_step=epoch)
                    tensorboardX_writer.add_scalar('val_acc', val_correct, global_step=epoch)

        # 如果三次都没有下降0.5的话就终止
        if val_loss > last_loss:
            patience = patience - 1
        else:
            patience = 5000
        last_loss = val_loss

        # 终止条件
        if patience == 0:
            break

    tensorboardX_writer.close()

    # # 开始训练
    # model.train()
    # best_loss = 99.999
    # for epoch in range(epoch_size):
    #
    #     right_num = 0
    #     train_loss = 0.0
    #     finally_loss = 0.0
    #
    #     loop = tqdm(database, total=len(database))
    #     for i, (inputs, labels) in enumerate(loop):
    #
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #
    #         optimizer.zero_grad()
    #
    #         predict = model(inputs)
    #         loss = loss_function(predict, labels)
    #
    #         # 常规的反向传播
    #         loss.backward()
    #         optimizer.step()
    #         # scheduler.step(finally_loss)
    #
    #         # 记录loss和命中数量
    #         train_loss += loss.item() * labels.shape[0]
    #         right_num += (predict.argmax(1) == labels).sum().item()
    #         finally_loss = train_loss/traces_num
    #         finally_acc = right_num/traces_num
    #
    #         # 显示进度条
    #         loop.set_description(f'[{epoch + 1}/{epoch_size}]')
    #         loop.set_postfix(loss=finally_loss, acc=finally_acc)
    #
    #     if epoch == 4:
    #         torch.save(model.state_dict(), 'epoch5/'+model_save_addr)
    #     if epoch == 5:
    #         torch.save(model.state_dict(), 'epoch6/'+model_save_addr)
    #     if epoch == 6:
    #         torch.save(model.state_dict(), 'epoch7/'+model_save_addr)
    #
    #     # 选择最好的loss进行保存
    #     if finally_loss <= best_loss:
    #         best_loss = finally_loss
    #         torch.save(model.state_dict(), model_save_addr)

    time_end = time.time()
    print('训练时长：', (time_end - time_start) / 60, '分')


def write_logs(addr, val):
    with open(addr, 'a') as f:
        f.write(str(val) + '\n')
