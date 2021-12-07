import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
import torch.nn.functional as F
from torchvision.utils import save_image


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')




#######################################################################################

#DEFINE A DISCRIMINATOR NETWORK

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 256
        ngpu = 1
        # input noise dimension
        nz = 100
        # number of generator filters
        ngf = 64
        # number of discriminator filters
        ndf = 64
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*2) x 16 x 16
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8

            nn.Conv2d(ndf, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),

            #nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print(input.shape)
        #if input.is_cuda and self.ngpu > 1:
        #    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        #else:
        output = self.main(input)
        #print(input.shape)
        #print(output.shape)

        return output.view(-1, 1).squeeze(1)

###########################################################################


def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    ###################################################################################################################
    # FAKE IMAGE GENERATOR i.e BNN
    logging.info("creating model fake generator (student)")
    model = models.__dict__['fgen2']
    model = model()
    #model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    #if args.model_config is not '':
    #    model_config = dict(model_config, **literal_eval(args.model_config))

    #model = model(**model_config)
    print(model)

    # cps = torch.load('results/student/model_best.pth.tar')
    # model.load_state_dict(cps)

    # REAL IMAGE GENERATOR i.e FP NETWORK
    logging.info("creating model real generator (student)")
    teacher = models.__dict__['rgen']
    model_config2 = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config2 = dict(model_config, **literal_eval(args.model_config))

    teacher = teacher()

    # cp = torch.load('results/teacher/model_best.pth.tar')['state_dict']
    cp = torch.load('../state_dicts/resnet34.pt')  # ['state_dict']
    teacher.load_state_dict(cp)

    discriminator = Discriminator().cuda()

    ###################################################################################################################

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = 400
            #best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint)
            #logging.info("loaded checkpoint '%s' (epoch %s)",
            #             checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    #regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
    #                                       'lr': args.lr,
    #                                       'momentum': args.momentum,
    #                                       'weight_decay': args.weight_decay}})
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    ###################################################################################################################
    ##### Sunjun Part ##########
    # define loss function (criterion) and optimizer

    criterion1 = nn.BCELoss()  ## Loss used for DCGAN
    criterion2 = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion3 = nn.KLDivLoss(reduction='batchmean')  ## Loss used for DCGAN
    # optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)  ## discriminator should be defined first
    criterion = criterion2
    # we can find any code on github
    # criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()



    criterion.type(args.type)
    criterion1.type(args.type)
    criterion2.type(args.type)
    criterion3.type(args.type)
    model.type(args.type)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=args.momentum)  # ,weight_decay=args.weight_decay)#, betas=(0.5, 0.999))
    optimizerD = torch.optim.SGD(discriminator.parameters(), lr=0.1, momentum=args.momentum,weight_decay=args.weight_decay)  # , betas=(0.5, 0.999))
    lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, last_epoch=args.start_epoch - 1, eta_min=0)
    lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.epochs,last_epoch=args.start_epoch - 1, eta_min=0)

    ## real label and fake label ##
    real_label = 1
    fake_label = 0

    ## setting device for the models ##
    teacher.cuda()
    teacher.eval()
    model.cuda()
    model.train()
    discriminator.cuda()

    ##### Sunjun Part ##########
    ###################################################################################################################


    for epoch in range(args.start_epoch, args.epochs):
        #optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader,  model, teacher, discriminator, criterion1, criterion, criterion3, epoch, lr_scheduler1, lr_scheduler2, optimizer, optimizerD,)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': args.model_config,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
             #'regime': regime
        }, is_best, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5)
        #results.plot(x='epoch', y=['train_loss', 'val_loss'],
        #             title='Loss', ylabel='loss')
        #results.plot(x='epoch', y=['train_error1', 'val_error1'],
        #             title='Error@1', ylabel='error %')
        #results.plot(x='epoch', y=['train_error5', 'val_error5'],
        #             title='Error@5', ylabel='error %')
        results.save()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
############################################################################################
def forward(data_loader, model, teacher, discriminator, criterion1, criterion, criterion3, epoch, lr_scheduler1, lr_scheduler2, training=True, optimizer=None, optimizer_D = None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    real_label = 1
    fake_label = 0
    device='cuda'

    teacher.eval()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()


        #else:
        input_var = Variable(inputs.type(args.type), volatile=not training)
        input_var.cuda()
        target_var = Variable(target)
        # compute output
        f_data, output = model(input_var)


        #r_data, rout = teacher(input_var)
        #r_data2, rout2 = teacher(input_var)
        label = torch.full((input_var.shape[0],), real_label, device=device).type(inputs.type()).cuda()




        r_data, rout = teacher(input_var)
        #f_data, output = model(input_var)

        kl_loss1 = 0.4*criterion3(F.log_softmax(output, dim=1), F.softmax(rout, dim=1))
        kl_loss2 = 0.4*criterion3(r_data, f_data)
        #kl_loss.backward()

        #'''


        #f_data, output = model(input_var)
        loss_real = 0.4*criterion(output, target_var)


        #print(loss_real.shape)
        #print(kl_loss.shape)


        loss = loss_real+kl_loss1#+kl_loss2

        #loss = loss_real
        if type(output) is list:
            output = output[0]


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            lr_scheduler1.step()
            lr_scheduler2.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            #print('[%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
            #    epoch, i, len(data_loader), kl_loss2.item(), kl_loss1.item(), D_x, D_G_z1, D_G_z2))
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

############################################################################################
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def forward2(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda()

        if not training:
            with torch.no_grad():
                input_var = Variable(inputs.type(args.type), volatile=not training)
                target_var = Variable(target)
                # compute output
                rdata, output = model(input_var)
        else:
            input_var = Variable(inputs.type(args.type), volatile=not training)
            target_var = Variable(target)
            # compute output
            rdata, output = model(input_var)



        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader,  model, teacher, discriminator, criterion1, criterion, criterion3, epoch, lr_scheduler1, lr_scheduler2, optimizer, optimizerD):
    # switch to train mode
    model.train()
    return forward(data_loader,  model, teacher, discriminator, criterion1, criterion, criterion3, epoch, lr_scheduler1, lr_scheduler2,
                   training=True, optimizer=optimizer, optimizer_D = optimizerD)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward2(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
