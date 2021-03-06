import argparse


def get_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--model_name', type=str, default="rcan")
    parser.add_argument('--discriminator_name', type=str, default="vgg19")
    parser.add_argument('--criterion_name', type=str, default="l1")
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size, default=64')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-3, help='select the learning rate, default=1e-3')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--adam', action='store_true', default=False, help='whether to use adam')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--save_model', action='store_true', default=True, help='enable to save pth')
    parser.add_argument('--save_model_pdf', action='store_true', default=False, help='enable to save the visual network')
    parser.add_argument('--seed', type=int, default=118, help="random seed")
    opt = parser.parse_args()

    return opt


def get_cyclegan_options(parser=argparse.ArgumentParser()):
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers, you had better put it '
                                                               '4 times of your gpu')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size, default=64')
    parser.add_argument('--batch_scale', type=int, default=4, help='input batch size, default=64')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for, default=100')
    parser.add_argument('--lr', type=float, default=1e-5, help='select the learning rate, default=1e-4')
    parser.add_argument('--adam', action='store_true', default=True, help='whether to use adam')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--seed', type=int, default=118, help="random seed")
    opt = parser.parse_args()

    return opt
