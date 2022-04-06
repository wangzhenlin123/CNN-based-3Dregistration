import numpy as np
import time
import torch
import os
from tqdm import tqdm
from model.DCN import *
import json



def load_checkpoint(model, checkpoint_PATH, optimizer):
    list_epoch = []
    for i in os.listdir(checkpoint_PATH):
        if ".tar" in i:
            list_epoch.append(int(i[6:-8]))
    load_epoch = np.max(list_epoch)
    checkpoint_PATH = checkpoint_PATH+"/epoch_"+str(load_epoch)+".pth.tar"

    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    epoch = model_CKPT['epoch'] + 1
    return model, optimizer, epoch


def train(start_epoch, net_kargs, model, device, save_path):
    for epoch in range(start_epoch, net_kargs['epoch']):
        model.train()
    
        epoch_total_loss = []
        epoch_step_time = []
        epoch_test_loss = []
    
        pbar = tqdm(range(0,net_kargs['train_iteration']))
        for step in pbar:
            pbar.set_description("train iteration %s" % step)
            step_start_time = time.time()

            a = torch.zeros(1,1,128,128,128)
            fixed_image, moving_image = dataloader #use your own DataProvider
            fixed_image = fixed_image.to(device)
            moving_image = moving_image.to(device)
            y_pred, matrix = model(moving_image, fixed_image, device)


            loss_fu = torch.nn.MSELoss()
            loss = loss_fu(y_pred,fixed_image)

            epoch_total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(train_loss_value=loss.item())
            epoch_step_time.append(time.time() - step_start_time)

        model.eval()
        with torch.no_grad():
            pbar = tqdm(range(0,net_kargs['test_iteration']))
            for step in pbar:
                fixed_image, moving_image = dataloader#use your own DataProvider
                fixed_image = fixed_image.to(device)
                moving_image = moving_image.to(device)
                y_pred, matrix = model(moving_image, fixed_image, device)

                loss_fu_test = torch.nn.MSELoss()
                loss_test = loss_fu(y_pred,fixed_image)

                epoch_test_loss.append(loss_test.item())
                pbar.set_postfix(test_loss_value=loss_test.item())

        if epoch % net_kargs['save_epoch'] == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           save_path + '/epoch_' + str(epoch) + '.pth.tar')

        epoch_info = 'Epoch %d/%d' % (epoch + 1, net_kargs['epoch'])
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)

        loss_info = 'loss: %.4e' % np.mean(epoch_total_loss)
        test_info = 'test_loss: %.4e' % np.mean(epoch_test_loss)

        print(' - '.join((epoch_info, time_info, loss_info,test_info)), flush=True)


if __name__ == '__main__':
    
    with open("config.json") as File:
        net_kargs = json.load(File)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load or initialize model
    if net_kargs['if_load'] == True:
        if os.listdir(net_kargs['save_path']) != []:
            model = DCN(net_kargs['input_channels'],32)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=net_kargs['lr'])
            model, optimizer, start_epoch= load_checkpoint(model, net_kargs['save_path'], optimizer)
    else:
        model = DCN(net_kargs['input_channels'],32)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=net_kargs['lr'])
        start_epoch = 0
    
    #train and save model
    train(start_epoch, net_kargs, model, device, net_kargs['save_path'])




