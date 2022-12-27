import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DistillationDataset(Dataset):
    #可能だったらこの中でデータセットを統合すべき？
    #Distillation用のデータセット
    #dataset...distillation_dataset.train_data
    #prob...各クラスである確率、それを2つ合わせたAveの高いものを各データのラベルとして採用する
    #count...Dataset生成回数、加重平均の重み

    def __init__(self, dataset, label1, label2, isinit):
        """init data

        Args:
            dataset (torch.Tensor): dataset
            label1 (list): target prob list
            label2 (tuple): picked prob list(random)
            isinit (bool): true if init dataset
        """
        if isinit:
            #init data
            
            self.dataset = dataset
            self.label = label1[0]
            self.prob = None
        else:
            self.dataset = dataset
            # print(len(dataset[0]))
            # print(type(label1))
            # # print(label1)
            # print(type(label1[0]))
            try:
                # print(len(label1[0]))
                # print(len(label2[0]))
                self.prob = culc_divide_list(label1, label2)
            except IndexError:
                print("!!Index Error!!")
                print(label1[0])
                print("len(label1[0]):", len(label1[0]))
                # print(label2)
                print(label2[0])
                print(label2[0][0])
                print(len(label2))
                print("len(label2[0]):", len(label2[0]))
                exit(0)
            # self.prob = [(label1[i]*self.count + label2[i])/(self.count+1) for i in range(len(label1))]
            # print(self.prob[0])
            # print(len(self.prob))
            # print("prob", self.prob[0])
            self.label = torch.tensor([torch.argmax(p) for p in self.prob])
            print(type(self.label[0]))
            print(self.label[0])
            # print("self.prob[0]:", self.prob[0:10])
            # print("self.label[0]:", self.label[0:10])
            # print("label:", self.label)
            # print("label[0] type:", self.label[0].dtype)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index], int(self.label[index])

def culc_divide_list(label1, label2):
    #label1...元のデータ
    #label2...統合するデータ
    #元のデータを加重平均
    # print("count:", n)
    for ll in label1[0]:
        assert not torch.isnan(ll).any()
    for ll in label2[0]:
        assert not torch.isnan(ll).any()
    new_label = [label1[0][i]*(label1[1]+1) + label2[0][i]*(label2[1]+1) for i in range(len(label1[0]))]
    # print("label1:", label1[0][0])
    # print("label2:", label2[0][0])
    res = []
    for ll in new_label:
        res.append(ll/(label1[1]+label2[1]))
    return res

def split_dataset(dataset, args):
    n_samples = len(dataset) # n_samples is 60000
    train_size = int(len(dataset) * args.dist_rate) # train_size is 48000
    
    train_dataset, distillation_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, (n_samples-train_size)], generator=torch.Generator().manual_seed(42))
    
    return train_dataset, distillation_dataset


def plot_multi_graph(data, args):
    #data...2次元リスト
    if args.synchronize:
        title = "synchronize_"
    else:
        title = "asynchronize_"
    if args.iid == 1:
        title += "Non-iid_pt1"
    elif args.iid == 2:
        title += "Non-iid_pt2"
    elif args.iid == 3:
        title += "Non-iid_pt3"
    elif args.iid == 4:
        title += "Non-iid_pt4"
    else:
        title += "IID"
    title +=  "_" + args.dataset
    title += f"_Model[{' '.join(args.model)}]"
    title += f"_lr[{args.lr}]" 
    title += f"_dist_rate[{args.dist_rate}]"
    data = np.array(data).T.tolist()
    x_range = len(data[0])
    # plt.figure(figsize=(4,3))
    for i, first in enumerate(data):
        plt.plot(range(1, 1+x_range), first, label = f"client {i+1}")
    plt.xlabel("round")
    plt.ylabel("Test Acc")
    if args.step3_epochs == 100:
        plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ["private", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"])
    else:
        plt.xticks([i for i in range(11)], ["private", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.savefig(f"fig/{title}.png")
    
def readable_size(size):
    for unit in ['K','M']:
        if abs(size) < 1024.0:
            return "%.1f%sB" % (size,unit)
        size /= 1024.0
    size /= 1024.0

def set_reindex(dataset, l):
    """
    dataset...torch.util.dataset.Subset
    l...index list
    """
    tmp = sorted(dataset.indices)
    res = []
    for ll in l:
        res.append(tmp.index(ll))
    return res

def float32_to_float16(data, size, device):
    res = torch.zeros(size, 10).to(device)
    for i in range(len(data)):
        for j in range(len(data[i])):
            #print(j, len(data[i]))
            res[i][j] = data[i][j].to(torch.float16)
    return res

def convert_3d_to_2d(l):
    res = []
    for i in range(len(l)):
        for j in range(len(l[i])):
            res.append(l[i][j])
    return res

def train(args, client_list, client, trainloader, dist_images, logger):
    # 各クライアント(！？)ごとに学習→蒸留を行う
    # 1回のみ
    # client_list[i].to(device)
    #if client % 2:
        #device = torch.device("cuda:1")
    #else:
        #device = torch.device("cuda:0")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = [], [] #各エポック数ごとのloss, accuracyで格納
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(client_list[client].parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_list[client].parameters(), lr=args.lr,
                                        weight_decay=1e-4)
        
        client_list[client].train()
    for epoch in range(args.step1_epochs):
        batch_loss = []
        acc_sum = 0
        loss_sum = 0
        num_train = 0
        for idx, (images, labels) in enumerate(trainloader):
            num_train += len(labels)
            # if idx == 0:
            #     print(type(images))
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = client_list[client](images)
            # if(epoch == 0 and idx == 0):
            #     print(outputs)
            #     print(len(outputs))
            # outputs = torch.softmax(outputs, dim=1)
            # print(type(outputs))
            #torch.tensor
            # print(type(labels))
            #torch.tensor
            # print(outputs)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            acc_sum += torch.sum(labels == torch.argmax(outputs, dim=1)).item()
            
            loss.backward()
            # print(loss.item())
            optimizer.step()

            # 学習状況の表示
            if (idx+1) % 25 == 0:
                logger.info(f"Device: {client+1}, Epoch: {epoch+1}/{args.step1_epochs}, Step: {idx+1}/{len(trainloader)}, Acc: {round(acc_sum*100 / num_train, 4)}%, Loss: {round(loss_sum/25, 4)}")
                batch_loss.append(loss_sum / 100)
                train_loss.append(sum(batch_loss)/len(batch_loss))
                loss_sum = 0
                
        train_acc.append(round(acc_sum / num_train, 4))
        train_loss.append(round(loss_sum/idx, 4))
        del images, labels
        torch.cuda.empty_cache()
    
    
    #推論部分
    client_list[client].eval()
    
    
    res = []
    with torch.no_grad():
        # print("distillationloader length:", len(distillationloader))
        for (images, labels) in dist_images:
            images = images.to(device)
            res.append(client_list[client](images)) #1回で追加されるのは0~9の確率が格納されたlist
            #distillation_client...250*48*10
    # print("res length:", len(res))
    
    
    return convert_3d_to_2d(res), train_acc, train_loss

def test(model, loader, logger):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    cor, total = 0, 0
    test_loss = 0.0
    num_test, acc_sum = 0, 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            num_test += len(labels)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            test_loss += criterion(output, labels).item()
            total += labels.size(0)
            acc_sum += int((labels == predicted).sum())
            del images, labels
            torch.cuda.empty_cache()
    
    test_loss /= len(loader)
            
    logger.info('Test Acc: {:.2f} % Loss: {:.2f}'.format(100 * round(acc_sum / num_test, 4), test_loss))
    return 100 * round(acc_sum / num_test, 4)
    
def dist_train(args, client_list, client, target, distillation_dataset, loader, logger):
    #distillation時
    #エポック1でトレードのみ行う
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = [], [] #各エポック数ごとのloss, accuracyで格納
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(client_list[client].parameters(), lr=args.lr,
                                    momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(client_list[client].parameters(), lr=args.lr,
                                        weight_decay=1e-4)
    epoch_loss = []
    batch_loss = []
    acc_sum = 0
    loss_sum = 0
    num_train = 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    distillation_loader = DataLoader(distillation_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    for idx, (images, labels) in enumerate(distillation_loader):
        # switch to half precission
        num_train += len(labels)
        # if idx == 0:
        #     print(type(images))
        torch.cuda.empty_cache()

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = client_list[client](images)
        # print(type(torch.from_numpy(np.array(labels)).long()))
        # print(type(outputs))
        if torch.isnan(outputs).any():
            # print("index:", index)
            print("!!outputs include nan!!")
            print("dataset:", images[0])
            print("len(dataset):", len(images))
            print("label:", labels[0])
            print("len(label):", len(labels))
            print("output:", outputs)
            print("len(output):", len(outputs))
            # print("model:", countZeroWeights(client_list[client]))
            check_datasets = int((images != images).sum())
            check_labels = int((labels != labels).sum())
            if(check_datasets>0):
                print("your data contains Nan")
            else:
                print("Your data does not contain Nan, it might be other problem")
            
            if(check_labels>0):
                print("your label contains Nan")
            else:
                print("Your label does not contain Nan, it might be other problem")
            
        loss = criterion(outputs, labels)
        assert not np.isnan(loss.item())
        loss_sum += loss.item()
        acc_sum += torch.sum(labels == torch.argmax(outputs, dim=1)).item()
        
        loss.backward()
        # switch to full precision
        
        # print(loss.item())
        optimizer.step()

        # 学習状況の表示
        if (idx+1) % 25 == 0:
            logger.info(f"From device {target+1} to target device {client+1}, Epoch: {1}/{args.epochs}, Step: {idx+1}/{len(distillation_loader)}, Acc: {round(acc_sum*100 / num_train, 4)}%, Loss: {round(loss_sum/25, 4)}")
            # batch_loss.append(loss_sum / 100)
            # epoch_loss.append(sum(batch_loss)/len(batch_loss))
            loss_sum = 0
        del images, labels
        torch.cuda.empty_cache()
    
    res = []
    with torch.no_grad():
        # print("distillationloader length:", len(distillationloader))
        for (images, labels) in loader:
            images = images.to(device)
            res.append(client_list[client](images)) #1回で追加されるのは0~9の確率が格納されたlist
    
    return convert_3d_to_2d(res)



def countZeroWeights(model):
    #デバッグ用
    #モデルの中の0の個数カウント
    zeros = 0
    for param in model.parameters():
        if param is not None:
            zeros += torch.sum((param == 0).int()).data[0]
    return zeros
