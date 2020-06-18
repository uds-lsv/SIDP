import argparse

parser = argparse.ArgumentParser(description='SIDPSGD')
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--noise_multiplier', type=float, default=7.1)
parser.add_argument('--clip', type=float, default=3.1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--normalization_type', default='Batch')
parser.add_argument('--device', default=None)
args = parser.parse_args()

if args.dataset == 'mnist':
     from src.mnist.train import main
elif args.dataset == 'cifar10':
     from src.cifar10.train import main

main(args.noise_multiplier, args.clip, args.lr, args.batch_size, args.epochs, args.normalization_type, args.device)

