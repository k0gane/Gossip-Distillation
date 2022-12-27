import sys
import os
import random
import time
from copy import deepcopy
from unicodedata import decimal
from cv2 import threshold
import torch
import datetime
import logging
from logging import getLogger
import warnings
import statistics

from torch.utils.data import Dataset, DataLoader
from src.models import get_densenet, get_resnet, get_mobilenet, get_ghostnet, get_effientnet
from src.utils import get_dataset, exp_details
from src.option import args_parser
from src.function import DistillationDataset, set_reindex, plot_multi_graph, readable_size, float32_to_float16, train, test, dist_train, culc_divide_list, countZeroWeights
from src.update import DatasetSplit

time.sleep(0.5)
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
# os.environ["TORCH_SHOW_CPP_STACKTRACES"] = '1'
args = args_parser()
warnings.simplefilter('ignore', UserWarning)
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO, filename=f"log/log_Synchronize[{args.synchronize}]_{args.dataset}_{args.num_users}_lr[{args.lr}]_IID[{args.iid}]_model[{' '.join(args.model)}]_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger = getLogger(__name__)
exp_details(args, logger)
#初期設定
if len(args.model) == 1: #ResNetだけ
    client_list = [get_resnet(True, args).to(device) for _ in range(args.num_users)]
else: #ResNet+なにか
    client_list = [get_resnet(True, args).to(device) for _ in range(args.num_users // 2)]
    another_model = args.model[1]
    for _ in range(args.num_users - (args.num_users // 2)):
        if another_model == "Dense":
            client_list.append(get_densenet().to(device))
        elif another_model == "mobilenet":
            client_list.append(get_mobilenet(args).to(device))
        #client_list.append(get_mobilenet(args).to(device))


train_dataset, distillation_dataset, test_dataset, user_groups = get_dataset(args)
distillation_dataset_images = [x[0] for x in distillation_dataset]
distillation_dataset_labels = [x[1] for x in distillation_dataset]
#distillation datasetは各ユーザーが共通で所持しているが、ラベルは異なってなければならない
distillation_datasets = [DistillationDataset(distillation_dataset_images, distillation_dataset_labels, None, True) for _ in range(args.num_users)]


#データ分割
trainloaders = []
for i in range(args.num_users):
    trainloaders.append(DataLoader(DatasetSplit(train_dataset, user_groups[i]), batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True))
    # if args.iid == 0:
    #     trainloaders.append(DataLoader(DatasetSplit(train_dataset, user_groups[i]), batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True))
    # else:
    #     trainloaders.append(DataLoader(DatasetSplit(train_dataset, set_reindex(train_dataset, user_groups[i])), batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True))

distillationloader = DataLoader(distillation_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

distillation_dataset_images_loader = distillationloader
distillation_dataset_images = [x[0] for x in distillation_dataset]
#推論に用いるloaderは統一のものにする



# criterion = CrossEntropyLoss2d()
distillation_result = {}
#distillation_result[0]...(result, count)

client_acc, client_loss, test_results = [], [], []
for client in range(args.num_users):
    # 各クライアント(！？)ごとに学習→蒸留を行う
    # 1回のみ
    distillation_result[client], ca, cl = train(args, client_list, client, trainloaders[client], distillation_dataset_images_loader, logger)
    distillation_result[client] = (float32_to_float16(distillation_result[client], len(distillation_dataset_images), device), 0)
    client_acc.append(ca)
    client_loss.append(cl)

    test_result = test(client_list[client], testloader, logger)
    test_results.append(test_result)

print("result of Each device:", test_results)
if len(args.model) == 1:
    print("median:", statistics.median(test_results))
    logger.info(f"median:{statistics.median(test_results)}")
else:
    print("median of ResNet:", statistics.median(test_results[:5]))
    print("median of other model:", statistics.median(test_results[:-5]))
    logger.info(f"median of ResNet:{statistics.median(test_results[:5])}")
    logger.info(f"median of other model:{statistics.median(test_results[:-5])}")

results = []
results.append(test_results)
#gossipでだいたいデータが行き渡るまでの期待値...log(n)
#pull
#...相手からデータを受信して自分のモデルを再学習
# Distillation step
# for count in range(math.ceil(math.log(args.num_users))):
enough_count = 0
threshold_acc = 80
test_results = []
distillation_result_for_send = deepcopy(distillation_result)
# print(distillation_result[0])
# print(distillation_result_for_send[0])
# print(type(distillation_result_for_send[0]))
if args.synchronize:
    for count in range(args.step3_epochs):
        for client in range(args.num_users):
            #重みの平均をとる
            target_list = [i for i in range(args.num_users) if i != client]
            target = random.choice(target_list)
            if count == 0 and client == 0:
                print(f"size of downloaded object is {readable_size(sys.getsizeof(distillation_result[target]))}.")
            distillation_datasets[client] = DistillationDataset(distillation_dataset_images, distillation_result[client], distillation_result_for_send[target], False)
            # print(len(distillation_dataset))
            # print(distillation_dataset[0])
            distillation_result[client] = (dist_train(args, client_list, client, target, distillation_dataset, distillation_dataset_images_loader, logger), distillation_result[client][1]+1)
            distillation_result_for_send[client] = deepcopy(distillation_result[client])
            test_result = test(client_list[client], testloader, logger)
            test_results.append(test_result)
            if test_result >= 90:
                enough_count += 1
        print("Step", count+1, "result:", test_results)
        if enough_count == args.num_users:
            print("Done!")
            results.append(test_results)
        else:
            # initialization
            enough_count = 0
            results.append(test_results)
            test_results = []
            
            
else:
    #非同期
    client_order = []
    results_count = [0 for _ in range(10)]
    for _  in range(args.step3_epochs):
        tmp = [i for i in range(10)]
        random.shuffle(tmp)
        client_order.extend(tmp)
        results.append([0 for i in range(args.num_users)])
        #client_order = [0...9が10個ずつランダムに並んでいる]
    random.shuffle(client_order)
    for count, client in enumerate(client_order):
        #重みの平均をとる
        target_list = [i for i in range(args.num_users) if i != client]
        target = random.choice(target_list)
        if count == 0:
            print(f"size of downloaded object is {readable_size(sys.getsizeof(distillation_result_for_send[target]))}.")
        
        # print(len(distillation_result[client]))
        # print(len(distillation_result_for_send[target]))
        distillation_datasets[client] = DistillationDataset(distillation_dataset_images, distillation_result[client], distillation_result_for_send[target], False)
        
        # print(len(distillation_dataset))
        # print(distillation_dataset[0])
        distillation_result[client] = (dist_train(args, client_list, client, target, distillation_dataset, distillation_dataset_images_loader, logger), distillation_result[client][1]+1)
        distillation_result_for_send[client] = deepcopy(distillation_result[client])
        test_result = test(client_list[client], testloader, logger)
        results[results_count[client]+1][client] = test_result
        results_count[client] += 1
        print("Step", count+1, "client:",client, "target:", target, "result:", results[results_count[client]][client])

if len(args.model) == 1:
    print("median:", statistics.median(results[-1]))
    logger.info(f"median:{statistics.median(results[-1])}")
else:
    print("median of ResNet:", statistics.median(results[-1][:5]))
    print("median of other model:", statistics.median(results[-1][:-5]))
    logger.info(f"median of ResNet:{statistics.median(results[-1][:5])}")
    logger.info(f"median of other model:{statistics.median(results[-1][:-5])}")
plot_multi_graph(results, args)
    # # 問題: 蒸留データセットの各xがどのクラスに属し、また各クライアントデータにどのクラスのセットが含まれるか利用することはできません。
        
        