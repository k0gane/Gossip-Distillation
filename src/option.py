import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users")
    # number of Epoch
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--step1_epochs', type=int, default=200,
                        help="number of rounds of local")
    parser.add_argument('--step3_epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--model', required=True, nargs="*", type=str,
                        help="enable multi model(resnet18 only or resnet18 and mobilenetv3)")
    parser.add_argument('--batch_size', type=int, default=40,
                        help="local batch size")
    parser.add_argument('--lr', type=float, default=0.000005,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--dist_rate', type=float, default=0.2,
                        help='Distillation Rate (default: 0.8)')
    # parser.add_argument('--num_channels', type=int, default=1, help="number \
    #                     of channels of imgs")#1...グレースケール、3...RGB

    # other arguments
    parser.add_argument('--block_rate', type=int, default=2, help="Synchronized Training or not.")

    parser.add_argument('--synchronize', type=bool, default=False, help="Synchronized Training or not.")
    parser.add_argument('--dataset', type=str, default='CINIC', help="name \
                        of dataset(MNIST or FMNIST or CIFAR or CINIC)")
    parser.add_argument('--optimizer', type=str, default='sgd', choices=["sgd", "adam"], 
                        help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to non-IID-1. 1-3 equal non-IID-1~3 and 0 for IID.')
    args = parser.parse_args()
    return args

