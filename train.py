from utils.augmentations    import SSDAugmentation
from layers.modules         import MultiBoxLoss
from ssd                    import build_ssd
from torch.autograd         import Variable
from data                   import *

import torch.backends.cudnn as cudnn
import torch.optim          as optim
import torch.utils.data     as data
import torch.nn.init        as init
import os.path              as osp
import torch.nn             as nn
import numpy                as np

import argparse
import torch
import time
import sys
import os



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# if called several times, write on the same line
def printf(string = ""):
    sys.stdout.write('\r'+string)

################################################################################ Arguments
parser    = argparse.ArgumentParser(
            description='Single Shot MultiBox Detector Training With Pytorch')
#train_set = parser.add_mutually_exclusive_group()

parser.add_argument('--dataset',               default='Personnal',           type=str,      help='VOC or COCO or Personnal',             choices=['VOC', 'COCO','Personnal'])
parser.add_argument('--batch_size',            default=16,                    type=int,      help='Batch size for training')
parser.add_argument('--resume',                default=None,                  type=str,      help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter',            default=0,                     type=int,      help='Resume training at this iter')
parser.add_argument('--num_workers',           default=4,                     type=int,      help='Number of workers used in dataloading')
parser.add_argument('--cuda',                  default=True,                  type=str2bool, help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3,                  type=float,    help='initial learning rate')
parser.add_argument('--momentum',              default=0.9,                   type=float,    help='Momentum value for optim')
parser.add_argument('--weight_decay',          default=5e-4,                  type=float,    help='Weight decay for SGD')
parser.add_argument('--gamma',                 default=0.1,                   type=float,    help='Gamma update for SGD')
parser.add_argument('--visdom',                default=False,                 type=str2bool, help='Use visdom for loss visualization')
parser.add_argument('--dataset_root',          default=P_ROOT,                               help='Dataset root directory path')
parser.add_argument('--val_dataset_root',      default=None,                                 help='Validation dataset root directory path. Only if dataset is "Personnal"')
parser.add_argument('--basenet',               default='vgg16_reducedfc.pth',                help='Pretrained base model')
parser.add_argument('--save_folder',           default='weights/',                           help='Directory for saving checkpoint models')
parser.add_argument('--early_stopping',        default=True,                  type=str2bool, help='Early stopping allow you to stop training when model is optimal for validation set')
parser.add_argument('--patience'               default=20,                    type=int,      help='Patience for early stopping, it corresponds to the number of epochs to wait a better results than the previous one. If there is no better model after the number of epochs given, the training stops')

args = parser.parse_args()

################################################################################ CUDA ou pas
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

################################################################################ Fonction d'entraînement
def train():

    ############################################################################ Chargement des datasets
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco # present in the /data/config.py
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc # present in the /data/config.py
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS))

    elif args.dataset == 'Personnal':
        cfg = personnal # present in the /data/config.py
        dataset     = PersonnalDetection(root=args.dataset_root,
                                         transform=SSDAugmentation(cfg['min_dim'], MEANS))
        val_dataset = PersonnalDetection(root=args.val_dataset_root,
                                         transform=SSDAugmentation(cfg['min_dim'], MEANS))

    ############################################################################ Visualisation des stats d'entraînement
    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ############################################################################ Création du réseau
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    ############################################################################ Reprendre entraînement
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    ############################################################################ Initialisastion des poids
    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    ############################################################################ Optimizer + Loss/criterion
    optimizer = optim.SGD(net.parameters(),
                          lr           = args.lr,
                          momentum     = args.momentum,
                          weight_decay = args.weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'],
                             0.5,
                             True,
                             0,
                             True,
                             3,
                             0.5,
                             False,
                             args.cuda)

    net.train()

    ############################################################################ Initialisation de paramètres
    # loss counters
    loc_loss   = 0
    conf_loss  = 0
    epoch      = 0

    print('Loading the dataset...')

    tmp        = len(dataset)
    tmp2       = tmp // args.batch_size
    epoch_size = tmp2 if tmp % args.batch_size == 0 else tmp2+1

    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    ############################################################################ Visualisation des stats d'entraînement
    if args.visdom:
        vis_title  = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot  = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch',     'Loss', vis_title, vis_legend)

    ############################################################################ Création du DataLoader
    data_loader     = data.DataLoader(dataset,
                                      args.batch_size,
                                      num_workers = args.num_workers,
                                      shuffle     = True,
                                      collate_fn  = detection_collate,
                                      pin_memory  = True)

    val_epoch_size   = len(val_dataset) // args.batch_size
    val_epoch_size_f = len(val_dataset) /  args.batch_size
    val_data_loader = data.DataLoader(val_dataset,
                                      args.batch_size,
                                      num_workers = args.num_workers,
                                      shuffle     = False,
                                      collate_fn  = detection_collate,
                                      pin_memory  = True)


    # create batch iterator
    batch_iterator = iter(data_loader)

    print("\n")
    if args.early_stopping:
        min_val_loss    = np.inf
        max_epochs_wait = args.patience
        epochs_wait     = 0
    waitToMuch      = False
    ############################################################################ Entraînement
    for iteration in range(args.start_iter, cfg['max_iter']):

        ######################################################################## Visualisation des stats d'entraînement
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch,
                            loc_loss,
                            conf_loss,
                            epoch_plot,
                            None,
                            'append',
                            epoch_size)

            # reset epoch loss counters
            loc_loss  = 0
            conf_loss = 0
            epoch    += 1

        ######################################################################## Modification du learning rate en fonction des itérations
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        ######################################################################## Chargement du batch
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator  = iter(data_loader)
            images, targets = next(batch_iterator)

        ######################################################################## Opitim Cuda
        if args.cuda:
            images  =  Variable(images.cuda())
            targets = [Variable(   ann.cuda(), volatile=True) for ann in targets]
        else:
            images  =  Variable(images)
            targets = [Variable(ann,           volatile=True) for ann in targets]

        ######################################################################## Propagation Avant
        # forward
        t0  = time.time()
        out = net(images)

        ######################################################################## Back Propagation
        # backprop
        optimizer.zero_grad()

        loss_l, loss_c = criterion(out, targets) # ref : forward() multibox_loss
        loss           = loss_l + loss_c

        loss.backward()
        optimizer.step()

        t1 = time.time()

        loc_loss  += loss_l.item()
        conf_loss += loss_c.item()

        ######################################################################## Affichage de stats d'entraînement
        if iteration % epoch_size == 0:
            #print('timer: %.4f sec.' % (t1 - t0))

            ####################################################################  CALCUL DE LA LOSS DE VALIDATION

            # create batch iterator
            val_batch_iterator = iter(val_data_loader)
            val_loss           = 0

            for i in range(val_epoch_size):
                try:
                    val_images, val_targets = next(val_batch_iterator)

                except StopIteration:
                    break

                if args.cuda:
                    val_images   =  Variable(images.cuda())
                    val_targets  = [Variable(ann.cuda()) for ann in val_targets]

                else:
                    val_images   =  Variable(val_images)
                    val_targets  = [Variable(ann) for ann in val_targets]

                detections   = net(val_images) # PENSER A NE PAS METTRE DE SHUFFLE

                optimizer.zero_grad()

                val_loss_l, val_loss_c = criterion(detections, val_targets) # ref : forward() multibox_loss

                val_loss              += val_loss_l.item() + val_loss_c.item()

                del val_images, val_targets, detections, val_loss_l, val_loss_c

            val_loss /= val_epoch_size_f

            print("")
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()) + ' ValLoss: %.4f ||' % (val_loss), end=' ')
            print("".join(["Iteration Globale: ", repr(iteration), " || Iteration Epoch: ", repr(iteration % epoch_size), "/", repr(epoch_size), " || Training Loss: %.4f" % (loss.item()), " || Validation Loss: %.4f" % (val_loss), " || Time per Iteration: %.4f sec." % (t1 - t0)]))

            """
            python train.py --cuda=False --num_workers=0 --batch_size=16 --dataset_root="/home/fnapierala/Bureau/Protocafeine/Detection/Dataset/train" --val_dataset_root="/home/fnapierala/Bureau/Protocafeine/Detection/Dataset/validation"
            """
            if args.early_stopping:
                if min_val_loss > val_loss:
                    epochs_wait            = 0
                    min_val_loss           = val_loss
                    min_val_loss_iteration = iteration
                    torch.save(ssd_net.state_dict(),
                               osp.join(args.save_folder,"".join([args.dataset,"_optimized.pth"])))
                else :
                    epochs_wait += 1
                    if epochs_wait >= max_epochs_wait:
                        waitToMuch = True
                        break
            if waitToMuch:
                break

        else :
            printf("".join(["Iteration Globale: ", repr(iteration), " || Iteration Epoch: ", repr(iteration % epoch_size), "/", repr(epoch_size), " || Training Loss: %.4f" % (loss.item())]))

        if waitToMuch:
            break

            #################################################################### End of modification

        if args.visdom:
            update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
                            iter_plot, epoch_plot, 'append')

        ######################################################################## Sauvegarde des poids toutes les 5000 itérations
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(),
                       args.save_folder + '' + args.dataset + repr(iteration) + '.pth')
            # torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
            #            repr(iteration) + '.pth')

    ############################################################################ Sauvegarde des poids du modèle final
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '_last.pth')

################################################################################ Ajustement du learning rate en fonction des itérations
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

################################################################################ Xavier initialisation
def xavier(param):
    init.xavier_uniform(param)

################################################################################ Initialisation des poids
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

################################################################################ Visualilsation visdom
def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

################################################################################ Visualisation visdom
def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
