import numpy as np
from torchvision import datasets, transforms

#prepare dataset for training
#MNIST: I.I.D, non-I.I.D, non-I.I.D(unequal amount)
#CIFAR-10: I.I.D, non-I.I.D
#色々と応用できそう
import numpy as np
from torchvision import datasets, transforms


def iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    #dict_users
    #all_idxs...0～47999
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users 

def noniid_pt1(dataset, num_users, args):
    """
    MNISTのラベルごとにフォルダを振り分ける
    0-1のみ, 2-3のみ, 4-5のみ,6-7のみ,8-9のみ
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    chase = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    # sort labels
    frag_list = [False for _ in range(5)]
    for data in dataset.indices:
        if args.dataset == "MNIST" or args.dataset == "FMNIST":
            idx = dataset.dataset.targets[data].item()
        elif args.dataset == "CIFAR":
            idx = dataset.dataset.targets[data]
        else:
            idx = dataset.dataset.data.targets[data]
        if frag_list[idx//2]:
            #初期か最後に入ったのが奇数
            dict_users[5+(idx//2)] = np.append(dict_users[5+(idx//2)], data)
            chase[5+(idx//2)] = np.append(chase[5+(idx//2)], idx)
        else:
            dict_users[(idx//2)] = np.append(dict_users[(idx//2)], data)
            chase[(idx//2)] = np.append(chase[(idx//2)], idx)
        frag_list[idx//2] = frag_list[idx//2] ^ True
    # for d in chase.values():
    #     print(len(d))
    #     print(d)
    #     print(len(np.bincount(d)))
    #     print(np.bincount(d))
    return dict_users


def noniid_pt2(dataset, num_users, args):
    """
    MNISTのラベルごとにフォルダを振り分ける
    (0-4, 5), (0-4, 6), (0-4, 7), (0-4, 8), (0-4, 9)
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    # sort labels
    frag = 0
    #0-4
    frag_list = [0 for _ in range(5)]
    #5-9
    for data in dataset.indices:
        if args.dataset == "MNIST" or args.dataset == "FMNIST":
            idx = dataset.dataset.targets[data].item()
        elif args.dataset == "CIFAR":
            idx = dataset.dataset.targets[data]
        else:
            idx = dataset.dataset.data.targets[data]
        if idx // 5:
            #5-9
            dict_users[5*frag_list[idx%5]+(idx%5)] = np.append(dict_users[5*frag_list[idx%5]+(idx%5)], data)
            frag_list[idx%5] = (frag_list[idx%5] + 1) % 2
        else:
            #0-4
            dict_users[frag] = np.append(dict_users[frag], data)
            frag =  (frag + 1) % 10
    return dict_users



def noniid_pt3(dataset, num_users, args):
    """
    MNISTのラベルごとにフォルダを振り分ける
    (0, 1, 2, 3), (0, 4, 5, 6), (1, 4, 7, 8), (2, 5, 7, 9), (3, 6, 8, 9)

    0...1, 2
    1...1, 3
    2...1, 4
    3...1, 5
    4...2, 3
    5...2, 4
    6...2, 5
    7...3, 4
    8...3, 5
    9...4, 5
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    # sort labels
    frag_dict = {
        0:(0, 1),
        1:(0, 2),
        2:(0, 3),
        3:(0, 4),
        4:(1, 2),
        5:(1, 3),
        6:(1, 4),
        7:(2, 3),
        8:(2, 4),
        9:(3, 4)
    }
    frag = 0
    frag_list = [0 for _ in range(5)] 
    #if args.dataset in ["MNIST", "FMNIST", "CIFAR"]:
    for data in dataset.indices:
        #print(dir(dataset.dataset.data))
        #print(dataset.dataset.data.targets)
        if args.dataset == "MNIST" or args.dataset == "FMNIST":
            idx = dataset.dataset.targets[data].item()
        elif args.dataset == "CIFAR":
            idx = dataset.dataset.targets[data]
        else:
            idx = dataset.dataset.data.targets[data]
        target = frag_dict[idx][frag] #今回のターゲット
        dict_users[5*frag_list[target]+target] = np.append(dict_users[5*frag_list[target]+target] , data)
        frag_list[target] = (frag_list[target] + 1) % 2
        frag = (frag + 1) % 2

    return dict_users

def noniid_pt4(dataset, num_users, args):
    """
    MNISTのラベルごとにフォルダを振り分ける
    0だけ, 1だけ, 2だけ, 3だけ...
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    # sort labels
    #if args.dataset in ["MNIST", "FMNIST", "CIFAR"]:
    for i, data in enumerate(dataset.indices): 
        if args.dataset == "MNIST" or args.dataset == "FMNIST":
            idx = dataset.dataset.targets[data].item()
        elif args.dataset == "CIFAR":
            idx = dataset.dataset.targets[data]
        else:
            idx = dataset.dataset.data.targets[data]
        dict_users[idx] = np.append(dict_users[idx], data)
    #else:
        #for i, data in enumerate(dataset):
            #idx = dataset.data.targets[i]
            #dict_users[idx] = np.append(dict_users[idx], i) 
    return dict_users

def noniid_pt5(dataset, num_users, args):
    """
    MNISTのラベルごとにフォルダを振り分ける
    偶奇で分ける
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    # sort labels
    frag_list = [0 for _ in range(2)]
    #if args.dataset in ["MNIST", "FMNIST", "CIFAR"]:
    for data in dataset.indices:
        #print(dir(dataset.dataset.data))
        #print(dataset.dataset.data.targets)
        if args.dataset == "MNIST" or args.dataset == "FMNIST":
            idx = dataset.dataset.targets[data].item()
        elif args.dataset == "CIFAR":
            idx = dataset.dataset.targets[data]
        else:
            idx = dataset.dataset.data.targets[data]
        if idx % 2:
            #奇数
            dict_users[2*frag_list[idx%2]+1] = np.append(dict_users[2*frag_list[idx%2]+1], data)
        else:
            #偶数
            dict_users[2*frag_list[idx%2]] = np.append(dict_users[2*frag_list[idx%2]], data)
        frag_list[idx%2] = (frag_list[idx%2] + 1) % 5

    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    # d = mnist_noniid(dataset_train, num)
    
# def dataset_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items,
#                                              replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users


# def dataset_noniid(dataset, num_users, num_shards, num_imgs):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     # num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users


# def mnist_noniid_unequal(dataset, num_users):
#     """
#     MNIST限定
#     non-IIDかつデータの量も不均一
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training imgs
#     """
    
#     """ A = tf.data.Dataset.range(10)
#         B = A.shard(num_shards=3, index=0)
#         list(B.as_numpy_iterator())
#         >>> [0, 3, 6, 9]
#         C = A.shard(num_shards=3, index=1)
#         list(C.as_numpy_iterator())
#         >>> [1, 4, 7]
#         D = A.shard(num_shards=3, index=2)
#         list(D.as_numpy_iterator())
#         >>> [2, 5, 8]
#     """
#     # 60,000 training imgs --> 50 imgs/shard X 1200 shards
#     num_shards, num_imgs = 1200, 50
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Minimum and maximum shards assigned per client:
#     min_shard = 1
#     max_shard = 30

#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard+1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)

#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:

#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         random_shard_size = random_shard_size-1

#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)
#     else:

#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#         if len(idx_shard) > 0:
#             # 残ったシャードを最小限の画像でクライアントに追加する:
#             shard_size = len(idx_shard)
#             # 残りのシャードをデータの少ないクライアントに追加する
#             k = min(dict_users, key=lambda x: len(dict_users.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[k] = np.concatenate(
#                     (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
#                     axis=0)

#     return dict_users


# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('./dataset/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,),
#                                                             (0.3081,))
#                                    ]))
#     num = 100
#     d = dataset_noniid(dataset_train, num)
