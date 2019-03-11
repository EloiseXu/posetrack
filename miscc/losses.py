import torch
import torch.nn as nn
from miscc.config import cfg

# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_maps, masks, fake_maps, conditions, real_labels, fake_labels, domain_labels):
    batch_size = fake_maps.size(0)
    num_joints = fake_maps.size(1)

    for idx in range(num_joints):
        for i in range(batch_size):
            fake_maps[i, idx] = fake_maps[i, idx] * masks[i, idx, 0]
            real_maps[i, idx] = real_maps[i, idx] * masks[i, idx, 0]

    real_features = netD(real_maps)
    fake_features = netD(fake_maps.detach())

    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)

    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss(fake_logits, fake_labels)

        domain_errD = 0
        if cfg.TRAIN.DOMAIN == True:
            domain_logits = netD.UNCOND_DNET(real_features)
            domain_errD = nn.BCELoss()(domain_logits, domain_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_wrong_errD) / 2.) + domain_errD
    else:
        errD = cond_real_errD + cond_wrong_errD
    return errD

def generator_loss(netsD, fake_maps, real_labels, img, masks, mpii):
    numDs = len(netsD)
    logs = ''

    batch_size = fake_maps.size(0)
    num_joints = fake_maps.size(1)

    errG_total = 0
    for i in range(numDs):
        for idx in range(num_joints):
            for j in range(batch_size):
                fake_maps[i, j, idx] = fake_maps[i, j, idx] * masks[j, idx, 0]
        features = netsD[i](fake_maps[i])
        cond_logits = netsD[i].COND_DNET(features, img)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss

        if cfg.TRAIN.DOMAIN == True:
            domain_fake_map = []
            real_domain_labels = []
            for j in range(batch_size):
                if mpii[j] == 0:
                    continue
                domain_fake_map.append(fake_maps[:, j])
                real_domain_labels.append(mpii[j])
            features = netsD[i].DOMAIN(domain_fake_map[i])
            logits = netsD[i].DOMAIN_DNET(features)
            errG_domain = nn.BCELoss()(logits, real_domain_labels)
            errG_total += errG_domain

        logs += 'g_loss%d: %.2f ' % (i, g_loss.data[0])

    return errG_total, logs