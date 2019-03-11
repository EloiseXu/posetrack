import os
import time
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.autograd import Variable

from miscc.config import cfg
from miscc.utils import weights_init
from miscc.losses import discriminator_loss, generator_loss

from model import hg
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, adjust_learning_rate
from pose.utils.osutils import join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import flip_back
import pose.losses as losses

class condGANTrainer(object):
    def __init__(self, train_loader, val_loader, args, njoints, device, idx):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_num_batches = len(self.train_loader)
        self.val_num_batches = len(self.val_loader)
        self.batch_size = args.train_batch
        self.num_classes = njoints
        self.device = device
        self.idx = idx
        self.epochs = args.epochs

        self.schedule = args.schedule
        self.gamma = args.gamma
        self.sigma_decay = args.sigma_decay
        self.checkpoint = args.checkpoint
        self.snapshot = args.snapshot
        self.num_stacks = args.stacks
        self.debug = args.debug
        self.flip = args.flip

        print("==> creating model '{}', stacks={}, blocks={}".format(args.arch,args.stacks, args.blocks))
        self.netG, self.netsD, self.start_epoch, self.best_acc, self.logger, self.optimizerG, self.optimizersD = \
                                      self.build_models(num_stacks=args.stacks,
                                      num_blocks=args.blocks,
                                      num_classes=njoints,
                                      resnet_layers=args.resnet_layers,
                                      device=device,
                                      args=args)
        self.real_labels, self.fake_labels = self.prepare_labels()

        self.criterion = losses.JointsMSELoss().to(device)

    def build_models(self, num_stacks, num_blocks, num_classes, resnet_layers, device, args):
        # #######################generator and discriminators############## #
        netsD = []
        from model import D_NET64 as D_NET
        netG = hg(num_stacks=num_stacks,
                  num_blocks=num_blocks,
                  num_classes=num_classes,
                  resnet_layers=resnet_layers)
        for i in range(num_stacks):
            netsD.append(D_NET())

        netG = torch.nn.DataParallel(netG).to(device)

        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        print('# of netsD', len(netsD))

        optimizerG, optimizersD = self.define_optimizers(netG, netsD)

        epoch = 0
        title = args.dataset + ' ' + args.arch
        best_acc = 0
        if cfg.TRAIN.NET_G != '':
            print('Load G from: ', cfg.TRAIN.NET_G)
            checkpoint = torch.load(cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            best_acc = checkpoint['best_acc']
            netG.load_state_dict(checkpoint['state_dict'])
            optimizerG.load_state_dict(checkpoint['optimizer'])
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = torch.load(Dname, map_location=lambda storage, loc:storage)
                    netsD[i].load_state_dict(state_dict)
        else:
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

        print('    Total params: %.2fM'
              % (sum(p.numel() for p in netG.parameters()) / 1000000.0))
        # ########################################################### #
        if cfg.CUDA:
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [netG, netsD, epoch, best_acc, logger, optimizerG, optimizersD]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()

        return real_labels, fake_labels

    def save_model(self, netsD, lr, epoch):
        valid_loss, valid_acc, predictions = self.validate()

        self.logger.append([epoch + 1, lr, valid_loss, valid_acc])

        is_best = valid_acc > self.best_acc
        best_acc = max(valid_acc, self.best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'best_acc': best_acc,
            'optimizer': self.optimizerG.state_dict(),
        }, predictions, is_best, checkpoint=self.checkpoint, snapshot=self.snapshot, epoch=epoch)

        for i in range(self.num_stacks):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                       '%s/netD%d.pth' % (self.checkpoint, i))
        print('Save G/Ds models.')

    def validate(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acces = AverageMeter()

        predictions = torch.Tensor(self.val_loader.dataset.__len__(), self.num_classes, 2)

        self.netG.eval()

        gt_win, pred_win = None, None
        end = time.time()
        bar = Bar('Eval ', max=len(self.val_loader))
        with torch.no_grad():
            for i, (input, target, meta, mpii) in enumerate(self.val_loader):
                if mpii == False:
                    continue
                data_time.update(time.time() - end)

                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                target_weight = meta['target_weight'].to(self.device, non_blocking=True)

                output = self.netG(input)
                score_map = output[-1].cpu() if type(output) == list else output.cpu()
                if self.flip:
                    flip_input = torch.from_numpy
                    flip_output = self.netG(flip_input)
                    flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                    flip_output = flip_back(flip_output)
                    score_map += flip_output

                if type(output) == list:
                    loss = 0
                    for o in output:
                        loss += self.criterion(o, target, target_weight)
                    output = output[-1]
                else:
                    loss = self.criterion(output, target, target_weight)

                acc = accuracy(score_map, target.cpu(), self.idx)

                preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
                for n in range(score_map.size(0)):
                    predictions[meta['index'][n], :, :] = preds[n, :, :]

                if self.debug:
                    gt_batch_img = batch_with_heatmap(input, target)
                    pred_batch_img = batch_with_heatmap(input, score_map)
                    if not gt_win or not pred_win:
                        plt.subplot(121)
                        gt_win = plt.imshow(gt_batch_img)
                        plt.subplot(122)
                        pred_win = plt.imshow(pred_batch_img)
                    else:
                        gt_win.set_data(gt_batch_img)
                        pred_win.set_data(pred_batch_img)
                    plt.pause(.05)
                    plt.draw()

                losses.update(loss.item, input.size(0))
                acces.update(acc[0], input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                        batch=i + 1,
                        size=len(self.val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc=acces.avg)

                bar.next()

            bar.finish()
        return losses.avg, acces.avg, predictions

    def main(self):
        batch_size = self.batch_size

        lr = cfg.TRAIN.GENERATOR_LR
        gen_iterations = 0
        for epoch in range(self.start_epoch, self.epochs):
            start_t = time.time()
            lr = adjust_learning_rate(self.optimizerG, epoch, lr, self.schedule, self.gamma)
            print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

            if self.sigma_decay > 0:
                self.train_loader.dataset.sigma *= self.sigma_decay
                self.val_loader.dataset.sigma *= self.sigma_decay

            self.train(gen_iterations, start_t, epoch, lr)

        self.logger.close()
        self.logger.plot(['Train Acc', 'Val Acc'])
        savefig(os.path.join(self.checkpoint, 'log.eps'))

        self.save_model(self.netsD, lr, self.epochs)

    def train(self, gen_iterations, start_t, epoch, lr):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.netG.train()

        end = time.time()

        gt_win, pred_win = None, None
        bar = Bar('Train', max=len(self.train_loader))
        step = 0
        errD_total = None
        errG_total = None
        for i, (input, target, meta, mpii) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            ######################################################
            # (1) Prepare training data and Compute text embeddings
            ######################################################
            input, target = input.to(self.device), target.to(self.device, non_blocking=True)
            target_weight = meta['target_weight'].to(self.device, non_blocking=True)

            #######################################################
            # (2) Generate fake heatmaps
            ######################################################
            output=self.netG(input)

            #######################################################
            # (3) Update D network
            ######################################################
            errD_total = 0
            D_logs = ''
            for i in range(self.num_stacks):
                self.netsD[i].zero_grad()
                errD = discriminator_loss(self.netsD[i], target, target_weight, output[i],
                                          input, self.real_labels, self.fake_labels, mpii)

                errD.backword()
                self.optimizersD[i].step()
                errD_total += errD
                D_logs += 'errD%d: %d.2f ' % (i,errD.data[0])

            #######################################################
            # (4) Update G network: maximize log(D(G(z)))
            ######################################################
            step += 1
            gen_iterations += 1

            self.netG.zero_grad()
            errG_total, G_logs = \
                generator_loss(self.netsD, output, self.real_labels, input, target_weight, mpii)

            if self.debug:
                gt_batch_img = batch_with_heatmap(input, target)
                pred_batch_img = batch_with_heatmap(input, output)
                if not gt_win or not pred_win:
                    ax1 = plt.subplot(121)
                    ax1.title.set_text('Groundtruth')
                    gt_win = plt.imshow(gt_batch_img)
                    ax2 = plt.subplot(122)
                    ax2.title.set_text('Prediction')
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            errG_total.backward()
            self.optimizerG.step()

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                    batch=i + 1,
                    size=len(self.train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td)
            bar.next()

            if gen_iterations % 100 == 0:
                print(D_logs + '\n' + G_logs)

        end_t = time.time()

        print('''[%d/%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.epochs,
                     errD_total.data[0], errG_total.data[0], end_t - start_t))

        if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
            self.save_model(self.netsD, lr, epoch)

        bar.finish()


