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
import torch.optim as optim
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
parser.add_argument('--epochs', default=250, type=int, metavar='N',
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
        nc = 512
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
            #nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            #nn.Conv2d(ndf, ndf * 2, 3, 1, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*2) x 16 x 16
            nn.Conv2d(nc, ndf, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf, ndf * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        #print(input.shape)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        #print(input.type())
        #print(output.type())

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






    ###################################################################################################################
    # FAKE IMAGE GENERATOR i.e BNN
    logging.info("creating model fake generator (student)")
    model = models.__dict__['fgen']
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    print(model)

    cps = torch.load('results/student/model_best.pth.tar')
    model.load_state_dict(cps)

    # REAL IMAGE GENERATOR i.e FP NETWORK
    logging.info("creating model real generator (student)")
    teacher = models.__dict__['rgen']
    model_config2 = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config2 = dict(model_config, **literal_eval(args.model_config))

    teacher = teacher(**model_config)

    cp = torch.load('results/teacher/model_best.pth.tar')['state_dict']
    teacher.load_state_dict(cp)


    discriminator = Discriminator().cuda()

    ###################################################################################################################

    logging.info("created model with configuration: %s", model_config)

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
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
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
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})



    ###################################################################################################################
    ##### Sunjun Part ##########
    # define loss function (criterion) and optimizer

    criterion = nn.BCELoss()  ## Loss used for DCGAN
    #criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()  ## Loss used for DCGAN
    #optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr)  ## discriminator should be defined first

    # we can find any code on github
    # criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)
    ##### Sunjun Part ##########
    ###################################################################################################################





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
        num_workers=args.workers, pin_memory=True, drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)

    #val_loss, val_prec1, val_prec5 = validate(
    #    val_loader, model.cuda(), criterion, 0)
    #return


    #for epoch in range(args.start_epoch, args.epochs):

    forward(args.batch_size, train_loader, args.epochs, teacher, model, discriminator, args, criterion, device = 'cuda')


###################################################################################################################
def forward(batch_size, data_loader, iteration_n, teacher_model, student_model, discriminator, args, criterion,device = 'cuda'):
        ## changed data_loader to batch_size and number of iteration ##
        ## iteration_n is number of iteration for one epoch, so iteration_n * batch_size = 1 epoch ##

        #### original code ####
        # if args.gpus and len(args.gpus) > 1:
        #     model = torch.nn.DataParallel(model, args.gpus)
        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        # losses = AverageMeter()
        # top1 = AverageMeter()
        # top5 = AverageMeter()
        #### original code ####


        ## optimizer ##
        optimizer_G = optim.SGD(student_model.parameters(), lr=0.01, momentum=args.momentum,weight_decay=args.weight_decay)#, betas=(0.5, 0.999))
        optimizer_D = optim.SGD(discriminator.parameters(), lr=0.1 , momentum=args.momentum, weight_decay=args.weight_decay)#, betas=(0.5, 0.999))
        lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, args.epochs,last_epoch=args.start_epoch - 1, eta_min=0)
        lr_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, args.epochs,last_epoch=args.start_epoch - 1, eta_min=0)

        ## real label and fake label ##
        real_label = 1
        fake_label = 0

        ## setting device for the models ##
        teacher_model.cuda()
        student_model.cuda()
        discriminator.cuda()

        ## noise dimension ##
        nd = 100

        for epoch in range(iteration_n):
            for i, (inputs, target) in enumerate(data_loader):

                inputs = inputs.to(device)
                ## initialization ##
                discriminator.zero_grad()

                ### defining noise for teacher model and student model ###
                noise_teacher = inputs.cuda()#torch.randn(batch_size, nd, 1, 1, device=device)
                noise_student = inputs.cuda()#torch.randn(batch_size, nd, 1, 1, device=device)

                ### generate real image and label ###
                r_data = teacher_model(noise_teacher)
                label = torch.full((batch_size,), real_label, device=device).type(inputs.type())

                ### training discriminator with real data ###
                output = discriminator(r_data)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                ### training discriminator with fake data ###
                f_data = student_model(noise_student)
                label.fill_(fake_label)
                output = discriminator(f_data.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD_total = errD_real + errD_fake
                optimizer_D.step()

                ### training student model (generator) ###
                student_model.zero_grad()
                label.fill_(real_label)
                output = discriminator(f_data)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizer_G.step()

                lr_scheduler1.step()
                lr_scheduler2.step()
                torch.save(student_model.state_dict(),"results/GAN_FD_BNN.pt")
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
                epoch, iteration_n, i, len(data_loader), errD_total.item(), errG.item(), D_x, D_G_z1, D_G_z2))


###################################################################################################################
def forward2(data_loader, model, criterion, epoch, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()



    #generate real image
    #real image training
    #generate fake image
    #fake image training


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
                output = model(input_var)
        else:
            input_var = Variable(inputs.type(args.type), volatile=not training)
            target_var = Variable(target)
            # compute output
            output = model(input_var)


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


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward2(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
