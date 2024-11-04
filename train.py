import sys
import time

sys.path.append('./')
from scipy import io
import os
import numpy as np
import torch
import argparse
from modules import dataset, TestNet, loss, transform
from utils import yaml_config_hook, save_model, metric, initialization_utils
from Toolbox import Preprocessing

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)


def train(model, loss_op, train_loader, optimizer):
    model.train()
    loss_epoch = 0
    for step, ((x_1, x_2), y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_list_1 = [x_i.to(DEVICE) for x_i in x_1]
        x_list_2 = [x_i.to(DEVICE) for x_i in x_2]
        y1, y2 = model(x_list_1, x_list_2)
        loss_, loss_con, loss_clu = loss_op(y1, y2, model.clustering_head.cluster_centers)
        # loss_, loss_con, loss_clu = loss_op(y1, y2)

        loss_.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t loss: "  f"{loss_.item():.6f}\t" f'CL:{loss_con.item():.6f}\t CLU: {loss_clu.item():.6f}')
        loss_epoch += loss_.item()
    return loss_epoch


def inference(test_loader, model, device, is_labeled_pixel):
    model.eval()
    y_pred_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(test_loader):
        x_list = [x_i.to(device) for x_i in x]
        with torch.no_grad():
            pred = model.forward_cluster(x_list)
        y_pred_vector.extend(pred.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 50 == 0:
            print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    y_pred_vector = np.array(y_pred_vector)     # 预测得到的聚类
    labels_vector = np.array(labels_vector)     # gt所得到的
    # print("Features shape {}".format(y_pred_vector.shape))
    if is_labeled_pixel:
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(labels_vector, y_pred_vector)
    else:
        indx_labeled = np.nonzero(labels_vector)[0]
        y = labels_vector[indx_labeled]
        y_pred = y_pred_vector[indx_labeled]
        acc, kappa, nmi, ari, pur, ca = metric.cluster_accuracy(y, y_pred)
    print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}'.format(acc, kappa, nmi, ari, pur))
    # GT = io.loadmat(gt_path)
    # gt = GT['GT']   # Trento
    # gt = GT['gt']   # MUUFL
    # gt = GT['berlin_gt']   # Berlin
    # Preprocessing.Processor().show_class_map(y_pred, y,gt)    # GroundTruth
    # Preprocessing.Processor().show_class_map(y_pred, indx_labeled,gt)

    return acc, kappa, nmi, ari, pur, ca


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建一个 ArgumentParser 对象，用于处理命令行参数。
    config = yaml_config_hook("config.yaml")    # 调用之前定义的 yaml_config_hook 函数，从配置文件 "config.yaml" 中加载配置，并将返回的配置存储在 config 变量中
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()  # 解析命令行参数，并将解析结果存储在 args 变量中。
    pretrain_path = args.model_path + '/pretrain'   # 根据解析得到的参数值构造预训练路径。
    joint_train_path = args.model_path + '/joint-train' # 根据解析得到的参数值构造联合训练路径。
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    if not os.path.exists(joint_train_path):
        os.makedirs(joint_train_path)
    initialization_utils.set_global_random_seed(seed=args.seed)

    root = args.dataset_root

    # prepare data
    if args.dataset == "Houston":
        im_1, im_2 = 'data_HS_LR', 'data_MS_HR'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "Trento":  # HSI=(166,600,63) Lidar=(166,600,2)
        im_1, im_2 = 'Trento-HSI', 'Trento-Lidar'
        gt_ = 'Trento-GT'   # (166,600)
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "Augsburg":
        im_1, im_2 = 'data_HS_LR', 'data_SAR_HR'
        gt_ = 'data_DSM'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "MUUFL":
        im_1, im_2 = 'HSI', 'LiDAR'
        gt_ = 'gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    elif args.dataset == "Berlin":
        im_1, im_2 = 'berlin_hsi', 'berlin_sar'
        gt_ = 'berlin_gt'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
    else:
        raise NotImplementedError
    gt_path = root + gt_ + '.mat'
    dataset_train = dataset.MultiModalDataset(gt_path, *img_path, patch_size=(args.image_size, args.image_size),
                                              transform=transform.Transforms(size=args.image_size),
                                              is_labeled=False)
    class_num = dataset_train.n_classes
    print('Processing %s ' % img_path[0])
    print(dataset_train.data_size, class_num)
    print(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # 表示当最后一个批次的样本数目不足一个完整的批量时，将其丢弃
        num_workers=args.workers,   # 指定用于数据加载的子进程数。通过使用多个子进程，数据加载器可以更高效地从数据集中加载样本，并有利于加快训练速度。
        prefetch_factor=4   # 预取因子，指定了数据加载器预取的倍数。预取可以减少数据加载时间和提高数据加载效率，特别是在数据加载和训练模型的计算步骤可以并行进行时。
    )

    # # test loader
    dataset_test = dataset.MultiModalDataset(gt_path, *img_path,
                                             patch_size=(args.image_size, args.image_size),
                                             transform=None, is_labeled=args.is_labeled_pixel)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=512,
                                                   shuffle=False,
                                                   drop_last=False, num_workers=args.workers)

    # initialize model
    model = TestNet.Net(dataset_train.n_modality, dataset_train.in_channels,
                        (args.image_size, args.image_size), 32, class_num, args.dim_emebeding)
    # print(model)
    # summary(model, (args.in_channel, rargs.image_size, args.image_size), device='cpu')
    model = model.to(DEVICE)

    # from thop import profile
    # inputs = [torch.randn(1, 63, 7, 7).to(DEVICE), torch.randn(1, 2, 7, 7).to(DEVICE)]
    # flops, params = profile(model, (inputs, inputs))
    # print('flops: ', flops, 'params: ', params)

    # optimizer / loss 对模型的不同部分应用不同的学习率。第一个字典元素中的参数用于普通训练，使用相同的学习率；而第二个字典元素中的参数用于聚类，使用不同的学习率
    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if 'clustering_head' not in n],
         'lr': args.learning_rate},  # # set SE layer
        {"params": model.clustering_head.cluster_centers, 'lr': args.learning_rate * args.lr_scale}
    ]
    optimizer = torch.optim.Adam(grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # # ===== joint training ==========
    score_list = []
    each_class = []
    acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, is_labeled_pixel=args.is_labeled_pixel)
    score_list.append([acc, kappa, nmi, ari, pur])
    print(f'initial accuracy: ACC={acc:.4f}')

    # save_model(joint_train_path, model, optimizer, 0)

    loss_op_joint = loss.JointLoss(args.batch_size,  # class_num,  #
                                   lambda_=args.contrastive_param,
                                   weight_clu=args.weight_clu_loss,
                                   regularization_coef=args.regularizer_coef, device=DEVICE)

    loss_history = []

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    print('start fine-tuning ...')
    start_time = time.time()
    for epoch in range(1, args.joint_train_epoch + 1):
        loss_epoch = train(model, loss_op_joint, data_loader_train, optimizer)
        print(f"Epoch [{epoch}/{args.joint_train_epoch}]\t Loss: {loss_epoch / len(data_loader_train)}")
        if epoch % 1 == 0:
            acc, kappa, nmi, ari, pur, ca = inference(data_loader_test, model, DEVICE, is_labeled_pixel=args.is_labeled_pixel)
            score_list.append([acc, kappa, nmi, ari, pur])
            each_class.append([ca])
            # save_model(joint_train_path, model, optimizer, epoch)
        loss_history.append(loss_epoch / len(data_loader_train))
        lr_scheduler.step()
    running_time = time.time() - start_time
    print(f'fine tuning time: {running_time:.3f} s')
    save_model(joint_train_path, model, optimizer, args.joint_train_epoch)
    print(loss_history)
    print(score_list)
    print(each_class)
