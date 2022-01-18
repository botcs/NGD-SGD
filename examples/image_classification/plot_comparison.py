# coding: utf-8
import torch
import glob
import os
import matplotlib.pyplot as plt

def kl(fname1, fname2):
    y1 = torch.load(fname1)
    y2 = torch.load(fname2)
    
    y1 = torch.nn.LogSoftmax(dim=1)(y1)
    y2 = torch.nn.LogSoftmax(dim=1)(y2)
    return torch.nn.functional.kl_div(y1, y2, reduction="batchmean", log_target=True)

def get_divs(dirname1, dirname2):
    fnames1 = glob.glob(f"{dirname1}/val_outputs_*.pth")
    fnames2 = glob.glob(f"{dirname2}/val_outputs_*.pth")
    fnames1.sort()
    fnames2.sort()
    divs = []
    for f1, f2 in zip(fnames1, fnames2):
        assert os.path.basename(f1) == os.path.basename(f2), (f1, f2)
        divs.append(kl(f1, f2))

    return divs

def get_self_comparison_divs(dirname):
    fnames = glob.glob(f"{dirname}/val_outputs_*.pth")
    fnames.sort()
    self_comparison_divs = []
    for f in fnames:
        self_comparison_divs.append(kl(f, fnames[-1]))

    return self_comparison_divs


if __name__ == "__main__":
    sgd_vs_sgdreparam_divs = get_divs(
        "exp/cifar/vs-seed42/SGD",
        "exp/cifar/vs-seed42/SGD-reparametrized"
    )

    ngd_vs_ngdreparam_divs = get_divs(
        "exp/cifar/vs-seed42/NGD",
        "exp/cifar/vs-seed42/NGD-reparametrized"
    )

    sgd_vs_ngd_divs = get_divs(
        "exp/cifar/vs-seed42/SGD",
        "exp/cifar/vs-seed42/NGD"
    )

    sgdreparam_vs_ngdreparam_divs = get_divs(
        "exp/cifar/vs-seed42/SGD-reparametrized",
        "exp/cifar/vs-seed42/NGD-reparametrized"
    )

    self_comparison_divs = get_self_comparison_divs(
        "exp/cifar/vs-seed42/SGD",
    )

    plt.close()
    plt.figure()
    plt.title("SEED 42")
    plt.plot(sgd_vs_sgdreparam_divs, label="SGD-vs-SGDreparam", alpha=.7)
    plt.plot(ngd_vs_ngdreparam_divs, label="NGD-vs-NGDreparam", alpha=.7)
    plt.plot(sgd_vs_ngd_divs, label="SGD-vs-NGD", alpha=.7)
    plt.plot(sgdreparam_vs_ngdreparam_divs, label="SGDreparam-vs-NGDreparam", alpha=.7)
    plt.plot(self_comparison_divs, label="SGD[:] vs SGD[-1]", alpha=.7)
    plt.legend()
    plt.savefig("test-42.png")
    


    plt.close()
    plt.figure()
    sgd_divs = get_divs(
        "exp/cifar/vs/SGD",
        "exp/cifar/vs-seed42/SGD",
    )

    ngd_divs = get_divs(
        "exp/cifar/vs/NGD",
        "exp/cifar/vs-seed42/NGD",
    )

    sgdreparam_divs = get_divs(
        "exp/cifar/vs/SGD-reparametrized",
        "exp/cifar/vs-seed42/SGD-reparametrized",
    )

    ngdreparam_divs = get_divs(
        "exp/cifar/vs/NGD-reparametrized",
        "exp/cifar/vs-seed42/NGD-reparametrized",
    )


    plt.title("SEED 0 vs 42")
    plt.plot(sgd_divs, label="SGD", alpha=.7)
    plt.plot(ngd_divs, label="NGD", alpha=.7)
    plt.plot(sgdreparam_divs, label="SGDreparam", alpha=.7)
    plt.plot(ngdreparam_divs, label="NGDreparam", alpha=.7)
    plt.legend()
    plt.savefig("test-0-vs-42.png")



    plt.title("Overall")
    plt.plot(sgd_divs, label="SGD", alpha=.7, color="grey")
    plt.plot(ngd_divs, label="NGD", alpha=.7, color="grey")
    plt.plot(sgdreparam_divs, label="SGDreparam", alpha=.7, color="grey")
    plt.plot(ngdreparam_divs, label="NGDreparam", alpha=.7, color="grey")
    plt.legend()
    plt.savefig("test-0-vs-42.png")