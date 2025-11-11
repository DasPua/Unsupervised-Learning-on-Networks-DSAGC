from collections import Counter
from sklearn.cluster import KMeans

import utils
from Upstream import GAE
from Downstream import MultiheadedGraphAttention
from evaluation import eva

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import csv


def trainer(dataset):
    Upstream = GAE(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(Upstream)
    Upstream.load_state_dict(torch.load(args.pretrain_path, map_location="cpu"))

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    with torch.no_grad():
        _, z = Upstream(data, adj, M)
    features = z.data.cpu().numpy()

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)  # (2708,)
    acc, nmi, ari, f1 = eva(y, y_pred, "pretrain-Kmeans", show=True)

    cluster = kmeans.cluster_centers_

    for i in range(y_pred.shape[0]):
        dis = []
        for j in range(cluster.shape[0]):
            dist = np.sqrt(np.sum(np.square(features[i] - cluster[j])))
            dis.append(dist)
        if not dis.index(min(dis)) == y_pred[i]:
            print("error")
        dis.clear()

    distances_samples = []
    for i in range(features.shape[0]):
        dist_temp = np.sum(np.square(features[i] - cluster[y_pred[i]]))
        distances_samples.append(dist_temp)
    distances_samples = np.array(distances_samples)

    alpha_k = 0.36  # 0.3
    realiable_sample = np.argwhere(distances_samples < alpha_k)

    y_new = [int(y[i]) for i in realiable_sample]
    y_pred_new = [int(y_pred[i]) for i in realiable_sample]
    eva(y_new, y_pred_new, "pretrain-ss", show=True)
    print(Counter(y_pred_new))
    print(Counter(y))

    train_self_supervised(
        x=data, y=y, adj=adj, y_pred=y_pred, sample_index=realiable_sample, epoch=400
    )


def train_self_supervised(x, y, adj, y_pred, sample_index, epoch=200):

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/Trainer_{args.name}.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_acc",
                "train_nmi",
                "train_ari",
                "train_f1",
                "test_acc",
                "test_nmi",
                "test_ari",
                "test_f1",
            ]
        )
    sample_index = [i[0] for i in sample_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, adj, y_pred = x.to(device), adj.to(device), torch.LongTensor(y_pred).to(device)

    model = MultiheadedGraphAttention(
        number_of_features=x.shape[1],
        number_of_hidden_layers=8,
        number_of_output_classes=int(y.max()) + 1,
        dropout=0.18,
        number_of_heads=8,
        alpha=0.2,
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-5)
    acc_best = 0
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        output = model(x, adj)
        loss_train = F.nll_loss(output[sample_index], y_pred[sample_index])
        loss_train.backward()
        optimizer.step()

        y_pre_GAT = np.argmax(output.cpu().detach().numpy(), axis=1)
        y_pre_kmeans = y_pred.cpu().detach().numpy()
        acc_train, nmi_train, ari_train, f1_train = eva(
            y_pre_GAT[sample_index], y_pre_kmeans[sample_index], "{}".format(i)
        )
        acc_test, nmi_test, ari_test, f1_test = eva(y_pre_GAT, y, "Test: {}".format(i))
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    i,
                    acc_train,
                    nmi_train,
                    ari_train,
                    f1_train,
                    acc_test,
                    nmi_test,
                    ari_test,
                    f1_test,
                ]
            )

        print(
            f"epoch - Train {i}:acc {acc_train:.4f}, nmi {nmi_train:.4f}, ari {ari_train:.4f}, f1 {f1_train:.4f}"
        )
        if acc_test > acc_best:
            print("the model outperformed the previous score : ")
            acc_best = acc_test
            torch.save(model.state_dict(), f"pretrain/self_GCN_{args.name}.pkl")
            best_output = output
            torch.save(best_output, "logs/best_output.pt")
            torch.save(y_pred, "logs/y_pred.pt")
            print(
                f"epoch - Train {i}:acc {acc_train:.4f}, nmi {nmi_train:.4f}, ari {ari_train:.4f}, f1 {f1_train:.4f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Citeseer")  # Citeseer, Cora
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--update_interval", default=1, type=int)  # [1,3,5]
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()

    if os.path.exists("pretrain/") == False:  #
        os.makedirs("pretrain/")

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == "Citeseer":
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None
    file_name = os.listdir("./pretrain")[0]
    args.pretrain_path = f"pretrain/GAE_{args.name}.pkl"
    args.input_dim = dataset.num_features

    print(args)
    trainer(dataset)
