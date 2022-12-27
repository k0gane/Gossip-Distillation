import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 各エポックごとに行う処理が入ってるやつ

class DatasetSplit(Dataset):
    """
    Pytorch Dataset クラスをラップした抽象 Dataset クラスです。
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object): #エポックごとのアップデート
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger #tensorboard.SumeryWriter
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        与えられたデータセットとユーザーインデックスに対して、
        train、validation、testの各データローダを返す
        """
        # train, validation, test用にインデックスを80:10:10に分割
        idxs_train = idxs[:int(0.8*len(idxs))] 
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_batch_size, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, args, model, model_count, federated_count):
        """
        各エポックごとの更新
        Args:
            args (sys.args): 引数
            model (nn.Module): global_modelのコピー
            global_round (int): 現在のラウンド数
            model_count (int): 現在のモデルの番号
            federated_count (int): 現在のFLの試行回数

        Returns:
            model.state_dict() 
            sum(epoch_loss) / len(epoch_loss): loss
        """
        # トレーニングモードへ変更
        model.train()
        epoch_loss = []

        # ローカルアップデートに用いるオプティマイザー(sgd or Adam)
        # https://pytorch.org/docs/stable/optim.html
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()#モデルのすべてのパラメータの勾配を0にする。
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Model Count : {} | Epoch : {} || [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        federated_count, model_count, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ 
        推論の精度と損失を返します。
        """

        model.eval()#推論モード
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(model, test_dataset):
    """ 
    テストの精度と損失を返します。
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
