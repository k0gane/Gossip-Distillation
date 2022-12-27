import torch
import numpy as np
from torchvision import datasets, transforms
from .function import split_dataset
from .sampling import iid, noniid_pt1, noniid_pt2, noniid_pt3, noniid_pt4
from pytorch_cinic.dataset import CINIC10
def get_dataset(args):
    """ 
    input:
        args.dataset (String)
        args.iid (Bool)   
    訓練データセットとテストデータセット，およびユーザグループを返します．
    キーはユーザーインデックス，値は各ユーザーに対応するデータです．各ユーザの対応するデータです。
    """

    if args.dataset == 'CIFAR':
        data_dir = '../data/cifar/'

        #transforms.Compose...前処理を行うための関数
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), #ndArray -> Tensor
            #↓ (R_Ave, G_Ave, B_Ave), (R_sgd, G_sgd, B_sgd)
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #正規化

        #data_dir...save dir
        #train...Trueならtrain, Falseならtest
        #download...dirにダウンロードする
        #transform...データセットに対する前処理
        tradiss_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        train_dataset, distillation_dataset = split_dataset(tradiss_dataset, args)

    elif args.dataset == 'CINIC':
        data_dir = '../data/cinic/'

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), #ndArray -> Tensor
            #↓ (R_Ave, G_Ave, B_Ave), (R_sgd, G_sgd, B_sgd)
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #正規化
        tradiss_dataset = CINIC10(data_dir, partition="train", download=True,
                                       transform=apply_transform)
        train_dataset, distillation_dataset = split_dataset(tradiss_dataset, args)
        test_dataset = CINIC10(data_dir, partition="test", download=True,
                                      transform=apply_transform)
    elif args.dataset == 'MNIST':
        data_dir = '../data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]) #mnistのAve, Stdがそんな感じらしい

        tradiss_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        train_dataset, distillation_dataset = split_dataset(tradiss_dataset, args)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    elif args.dataset == 'FMNIST':
        data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        tradiss_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        train_dataset, distillation_dataset = split_dataset(tradiss_dataset, args)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # sample training data amongst users
    if args.iid==1:
        # Sample IID user data from Mnist
        user_groups = noniid_pt1(train_dataset, args.num_users, args)
    elif args.iid==2:
        user_groups = noniid_pt2(train_dataset, args.num_users, args)
    elif args.iid==3:
        user_groups = noniid_pt3(train_dataset, args.num_users, args)
    elif args.iid==4:
        user_groups = noniid_pt4(train_dataset, args.num_users, args)
    else:
        user_groups = iid(train_dataset, args.num_users)
    # print(user_groups)
    if args.iid >= 1:
        train_dataset = tradiss_dataset
    return train_dataset, distillation_dataset, test_dataset, user_groups

def exp_details(args, logger):
    '''
    学習内容の出力
    '''
    
    logger.info('Experimental details:')
    logger.info(f'    Number of Users : {args.num_users}')
    logger.info(f'    Epochs     : {args.epochs}')
    logger.info(f'    Optimizer : {args.optimizer}')
    logger.info(f'    Learning Rate : {args.lr}')
    logger.info(f'    Model : {" ".join(args.model)}')
    logger.info(f'    Dataset : {args.dataset}')
    if args.iid == 1:
        logger.info('    Non-IID#1')
    elif args.iid == 2:
        logger.info('    Non-IID#2')
    elif args.iid == 3:
        logger.info('    Non-IID#3')
    elif args.iid == 4:
        logger.info('    Non-IID#4')
    else:
        logger.info('    IID')
    logger.info(f'    Batch size   : {args.batch_size}')

