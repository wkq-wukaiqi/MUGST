import os
import argparse
from pprint import pprint
import time
import random
import shutil
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from models import MUGST
from utils.logger import CompleteLogger
from utils.meter import AverageMeter, ProgressMeter
from utils.dataloader import get_dataloader
from utils.metrics import masked_mae, metric_tensor
from utils.visualize import plot_adj
from sklearn.cluster import KMeans


def discovery_community(adj, k):
    A = torch.tensor(adj, dtype=torch.float)
    # 计算度矩阵 D
    D = torch.diag(A.sum(dim=1))
    # 计算未归一化的拉普拉斯矩阵 L
    L = D - A
    # 将 L 转换为 NumPy 数组以使用 NumPy/SciPy 的特征值计算
    L_np = L.numpy()

    # 计算 L 的特征值和特征向量
    _, eigvecs = np.linalg.eigh(L_np)

    # 选择前 k 个最小的非零特征向量（假设 k 是社区数量）
    embedding = eigvecs[:, :k]

    # 使用 KMeans 聚类进行社区划分
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(embedding)
    return labels


def get_transfer_matrix(community_labels, num_nodes, num_community, device):
    # [N, M]
    t = torch.zeros((num_nodes, num_community), device=device)
    # t = torch.zeros(N, M, dtype=torch.int)

    # 使用 scatter_ 方法将指定的位置设置为 1
    t[torch.arange(num_nodes), community_labels] = 1
    return t.float()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [args.gpu]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_env(args.seed)
    logger = CompleteLogger(args.log, args.phase)

    pprint(vars(args))

    device = torch.device(device)

    train_loader, val_loader, test_loader, data_scaler, adj_mx = get_dataloader(args)
    num_nodes = adj_mx.shape[0]
    if args.spe_cluster:
        community_labels = discovery_community(adj_mx, k=args.num_coarse)
        community_labels = torch.tensor(community_labels, dtype=torch.long).to(device)
        transfer_matrix = get_transfer_matrix(community_labels, num_nodes, args.num_coarse, device)
    else:
        transfer_matrix = None

    model = MUGST(args, num_nodes=num_nodes, transfer_matrix=transfer_matrix)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_steps = [int(i) for i in list(args.lr_steps.split(','))]
    lr_scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=args.lr_gamma)

    loss_fun = masked_mae

    print("start training...")

    best_mae = float('inf')
    best_epoch = 0
    early_stopping_count = 0
    train_times = AverageMeter('Training Time', ':.4f')
    inference_times = AverageMeter('Inference Time', ':.4f')

    for epoch in range(1, args.epochs + 1):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # ==================== Train ====================
        training_time, adj = train(epoch, model, loss_fun, optimizer,
                                   train_loader, data_scaler, device, args)
        train_times.update(training_time, 1)
        lr_scheduler.step()
        # ==================== Validate ====================
        mae_val, rmse_val, mape_val, inference_time = validate(model, val_loader, data_scaler,
                                                               device, args)

        torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
        if mae_val < best_mae:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
            best_epoch = epoch
            early_stopping_count = 0
        else:
            early_stopping_count += 1
        best_mae = min(mae_val, best_mae)

        print(
            f'Valid Loss: {mae_val:.4f}, RMSE: {rmse_val:.4f}, MAPE: {mape_val:.4f}, best epoch is {best_epoch}, best loss is {best_mae:.4f}')
        print(f'Inference Time: {inference_time:.4f} secs')
        inference_times.update(inference_time, 1)
        if early_stopping_count >= args.patience:
            print(f'Validation performance didn\'t improve for {args.patience} epochs, early stopping')
            break

        if args.coarse and (not args.spe_cluster):
            transfer_matrix = model.transfer_matrix
            transfer_matrix = transfer_matrix.detach().cpu().numpy()
            pickle.dump(transfer_matrix, open(logger.get_pickle_path('transfer_matrix'), 'wb'))
            plot_adj(transfer_matrix, logger.get_image_path('transfer_matrix'))
            plot_adj(transfer_matrix[:50, :50], logger.get_image_path('transfer_matrix_part'))

        if args.sem_gconv and args.coarse:
            adj = adj.detach().cpu().numpy()
            pickle.dump(adj, open(logger.get_pickle_path(f'sematic_adj'), 'wb'))
            plot_adj(adj, logger.get_image_path(f'sematic_adj'))
            plot_adj(adj[:50, :50], logger.get_image_path(f'sematic_adj_part'))

    print("Training finished")
    print(f"The valid loss on best model is {str(round(best_mae, 4))}, best epoch is {best_epoch}")
    print("Average Training Time: {:.4f} secs/epoch".format(train_times.avg))
    print("Average Inference Time: {:.4f} secs".format(inference_times.avg))

    model.load_state_dict(torch.load(logger.get_checkpoint_path('best'), map_location=device))
    # ==================== Test ====================
    test(model, test_loader, data_scaler, device, args)

    logger.close()


def train(epoch, model, loss_fun, optimizer, train_loader, data_scaler, device, args):
    t1 = time.time()

    losses = AverageMeter('Loss-total', ':.4f')
    log_list = [losses]

    progress = ProgressMeter(
        len(train_loader),
        log_list,
        prefix="Epoch: [{}](MODEL)".format(epoch))

    model.train()

    for iter, (x, y) in enumerate(train_loader):
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)
        y = y[:, :, :, range(args.output_dim)]

        optimizer.zero_grad()

        output, adj = model(x)
        pred = data_scaler.inverse_transform(output)
        true = data_scaler.inverse_transform(y)
        loss = loss_fun(pred, true)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

        loss_value = loss.item()

        losses.update(loss_value, x.size(0))

        if iter % args.print_every == 0:
            progress.display(iter)

    t2 = time.time()
    training_time = t2 - t1
    print(f'Training Time: {training_time:.4f}/epoch')
    return training_time, adj


def validate(model, val_loader, data_scaler, device, args):
    progress = ProgressMeter(
        len(val_loader),
        [],
        prefix="Validate: ")

    model.eval()

    total_mae, total_mse, total_mape = 0, 0, 0
    total_samples = 0

    s1 = time.time()
    with torch.no_grad():
        for iter, (x, y) in enumerate(val_loader):
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            y = y[:, :, :, range(args.output_dim)]

            output = model(x)

            pred = output.detach()
            true = y.detach()

            pred = data_scaler.inverse_transform(pred)
            true = data_scaler.inverse_transform(true)

            mae, mse, mape = metric_tensor(pred, true)

            batch_size = pred.shape[0]
            total_mae += mae * batch_size
            total_mse += mse * batch_size
            total_mape += mape * batch_size
            total_samples += batch_size

            if iter % args.print_every == 0:
                progress.display(iter)

    s2 = time.time()
    inference_time = s2 - s1

    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples
    avg_mape = total_mape / total_samples
    avg_rmse = np.sqrt(avg_mse)

    return avg_mae, avg_rmse, avg_mape, inference_time


def test(model, test_loader, data_scaler, device, args):
    print('Testing...')

    progress = ProgressMeter(
        len(test_loader),
        [],
        prefix="Test: ")

    model.eval()

    total_mae = np.zeros((args.pred_len, args.output_dim))
    total_mse = np.zeros((args.pred_len, args.output_dim))
    total_mape = np.zeros((args.pred_len, args.output_dim))
    total_samples = 0

    with torch.no_grad():
        for iter, (x, y) in enumerate(test_loader):
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            y = y[:, :, :, range(args.output_dim)]

            output = model(x)

            pred = output.detach()
            true = y.detach()

            pred = data_scaler.inverse_transform(pred)
            true = data_scaler.inverse_transform(true)

            batch_size = pred.shape[0]
            total_samples += batch_size

            for d in range(args.output_dim):
                for i in range(args.pred_len):
                    pred_slice = pred[:, i, :, d]
                    true_slice = true[:, i, :, d]
                    mae, mse, mape = metric_tensor(pred_slice, true_slice)

                    # Accumulate the metrics
                    total_mae[i, d] += mae * batch_size
                    total_mse[i, d] += mse * batch_size
                    total_mape[i, d] += mape * batch_size

            if iter % args.print_every == 0:
                progress.display(iter)

    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples
    avg_mape = total_mape / total_samples
    avg_rmse = np.sqrt(avg_mse)

    print('*' * 60)
    print('For grid-based datasets, output contains 2 dims, means inflow and outflow, respectively')
    for d in range(args.output_dim):
        print('*' * 60)
        print(f'Evaluating output dim: [{d}]')

        amae, armse, amape = [], [], []
        for i in range(args.pred_len):
            log = 'Horizon {:d} Test MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'
            print(log.format(i + 1, avg_mae[i, d], avg_rmse[i, d], avg_mape[i, d]))
            amae.append(avg_mae[i, d])
            armse.append(avg_rmse[i, d])
            amape.append(avg_mape[i, d])

        log = 'On average over {:d} horizons, Test MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}'
        print(log.format(args.pred_len, np.mean(amae), np.mean(armse), np.mean(amape)))
        print('*' * 60)


def set_env(seed):
    # reproducibility
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")  # 仅单卡
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--log', type=str, default='logs/pems_bay', help='log path')
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis', 'debug'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    # dataset
    parser.add_argument('--dataset_dir', type=str, default='./datasets', help='root path of data')
    parser.add_argument('-d', '--dataset', type=str, help='')
    parser.add_argument("--dow", type=bool, default=True, help='Add feature day_of_week.')
    parser.add_argument("--tod", type=bool, default=True, help='Add feature time_of_day.')
    parser.add_argument("--steps_per_day", type=int, default=288)
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--num_workers', type=int, default=12, help='data loader num workers')
    # training strategy
    parser.add_argument('--learning_rate', type=float, default=0.002, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--resume_path', default=None, type=str)
    parser.add_argument('--epochs', type=int, default=200, help='')
    parser.add_argument('--lr_steps', type=str, default='1,40,80,120,160', help='parameter for lr scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='parameter for lr scheduler')
    parser.add_argument('--patience', type=int, default=30, help='patience for early stopping')

    # model
    parser.add_argument('--input_dim', type=int, default=1, help='input dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='output dimension')
    parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--num_layer', type=int, default=6, help='number of layers')
    parser.add_argument('--spatial_dim', type=int, default=64, help='node dimension')
    parser.add_argument('--temp_dim_tid', type=int, default=32, help='temporal dimension for T_i_D')
    parser.add_argument('--temp_dim_diw', type=int, default=32, help='temporal dimension for D_i_W')
    parser.add_argument('--day_of_week_size', type=int, default=7, help='day of week size')
    parser.add_argument('--gcn_dim', type=int, default=16, help='')
    parser.add_argument('--num_coarse', type=int, default=32, help='number of coarse-grained nodes')

    # ablation
    parser.add_argument('--spe_cluster', action='store_true', help='')
    parser.add_argument('--sem_gconv', action='store_true', help='')
    parser.add_argument('--fine', action='store_true', help='')
    parser.add_argument('--coarse', action='store_true', help='')

    args = parser.parse_args()
    main(args)
