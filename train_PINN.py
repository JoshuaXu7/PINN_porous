import torch
import torch.optim as optim
from models.codec import DenseED
from models.darcy import conv_constitutive_constraint as constitutive_constraint
from models.darcy import conv_continuity_constraint as continuity_constraint
from models.darcy import conv_boundary_condition as boundary_condition
from utils.image_gradient import SobelFilter
from utils.load import load_data
from utils.misc import mkdirs, to_numpy
from utils.plot_new import plot_prediction_det, plot_prediction_det2, save_stats
from utils.practices import OneCycleScheduler, adjust_learning_rate, find_lr
import time
import argparse
import random
from pprint import pprint
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Parser(argparse.ArgumentParser):
    def __init__(self): 
        super(Parser, self).__init__(description='Learning surrogate with mixed residual norm loss')
        self.add_argument('--exp-name', type=str, default='FVM_lrstep095_w10_600_10e14_newplot/mixed_residual', help='experiment name')
        self.add_argument('--exp-dir', type=str, default="./experiments", help='directory to save experiments')      
        # codec
        self.add_argument('--blocks', type=list, default=[6, 8, 6], help='list of number of layers in each dense block')# dense block
        self.add_argument('--growth-rate', type=int, default=16, help='number of output feature maps of each conv layer within each dense block')
        self.add_argument('--init-features', type=int, default=48, help='number of initial features after the first conv layer') # initial feature map     
        self.add_argument('--drop-rate', type=float, default=0., help='dropout rate')
        self.add_argument('--upsample', type=str, default='nearest', choices=['nearest', 'bilinear'])
        # data 
        self.add_argument('--data-dir', type=str, default="./datasets", help='directory to dataset')
        self.add_argument('--data', type=str, default='aniso', choices=['aniso', 'volume20','volume30', 'volume40','volume50', 'volume60','mixvolume']) # dataset
        self.add_argument('--ntrain', type=int, default=16384, help="number of training data")# number of samples in training data
        self.add_argument('--ntest', type=int, default=512, help="number of validation data")# number of samples in validation data
        self.add_argument('--imsize', type=int, default=128) # image pixel
        # training
        self.add_argument('--run', type=int, default=1, help='run instance')
        self.add_argument('--epochs', type=int, default=600, help='number of epochs to train')
        self.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.add_argument('--lr-div', type=float, default=2., help='lr div factor to get the initial lr')
        self.add_argument('--lr-pct', type=float, default=0.3, help='percentage to reach the maximun lr, which is args.lr')
        self.add_argument('--weight-decay', type=float, default=0., help="weight decay")
        self.add_argument('--weight-bound', type=float, default=10, help="weight for boundary loss")
        self.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
        self.add_argument('--test-batch-size', type=int, default=512, help='input batch size for testing')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--cuda', type=int, default=1, choices=[0, 1, 2, 3], help='cuda index')
        # logging
        self.add_argument('--debug', action='store_true', default=False, help='debug or verbose')
        self.add_argument('--ckpt-epoch', type=int, default=None, help='which epoch of checkpoints to be loaded')
        self.add_argument('--ckpt-freq', type=int, default=100, help='how many epochs to wait before saving model')
        self.add_argument('--log-freq', type=int, default=1, help='how many epochs to wait before logging training status')
        self.add_argument('--plot-freq', type=int, default=50, help='how many epochs to wait before plotting test output')
        self.add_argument('--plot-fn', type=str, default='contourf', choices=['contourf', 'imshow'], help='plotting method')

    def parse(self):
        args = self.parse_args()

        hparams = f'{args.data}_ntrain{args.ntrain}_run{args.run}_bs{args.batch_size}_lr{args.lr}_epochs{args.epochs}'
        if args.debug:
            hparams = 'debug/' + hparams
        args.run_dir = args.exp_dir + '/' + args.exp_name + '/' + hparams
        args.ckpt_dir = args.run_dir + '/checkpoints'
        mkdirs(args.run_dir, args.ckpt_dir) 

        assert args.ntrain % args.batch_size == 0 and \
            args.ntest % args.test_batch_size == 0 

        if args.seed is None:
            args.seed = random.randint(1, 10000) 
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4) 

        return args


if __name__ == '__main__':

    args = Parser().parse()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    args.train_dir = args.run_dir + '/training' 
    args.pred_dir = args.train_dir + '/predictions' 
    mkdirs(args.train_dir, args.pred_dir)

    model = DenseED(in_channels=1, out_channels=3, 
                    imsize=args.imsize,
                    blocks=args.blocks,
                    growth_rate=args.growth_rate, 
                    init_features=args.init_features,
                    drop_rate=args.drop_rate,
                    out_activation=None,
                    upsample=args.upsample)
    if args.debug:
        print(model)

    # if start from ckpt
    if args.ckpt_epoch is not None:
        ckpt_file = args.run_dir + f'/checkpoints/model_epoch{args.ckpt_epoch}.pth'
        model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        print(f'Loaded ckpt: {ckpt_file}')
        print(f'Resume training from epoch {args.ckpt_epoch + 1} to {args.epochs}')
    model = model.to(device)

    # load data
    if args.data == 'aniso':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/FVM_aniso_ext128pixel_n19680_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/FVM_aniso_ext128pixel_n574_test.hdf5'
        ntrain_total, ntest_total = 19680, 512 
    elif args.data == 'volume20':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume20_64pixel_n4100_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume20_64pixel_n512_test.hdf5'
        ntrain_total, ntest_total = 4100, 512 
    elif args.data == 'volume30':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume30_64pixel_n4100_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume30_64pixel_n512_test.hdf5'
        ntrain_total, ntest_total = 4100, 512 
    elif args.data == 'volume40':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume40_64pixel_n4100_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume40_64pixel_n512_test.hdf5'
        ntrain_total, ntest_total = 4100, 512 
    elif args.data == 'volume50':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume50_64pixel_n4100_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume50_64pixel_n512_test.hdf5'
        ntrain_total, ntest_total = 4100, 512 
    elif args.data == 'volume60':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume60_64pixel_n4100_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/volume60_64pixel_n512_test.hdf5'
        ntrain_total, ntest_total = 4100, 512 
    elif args.data == 'mixvolume':
        train_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/mixvolume_64pixel_n9840_train.hdf5'
        test_hdf5_file = args.data_dir + \
            f'/{args.imsize}x{args.imsize}/mixvolume_64pixel_n1148_test.hdf5'
        ntrain_total, ntest_total = 9840, 1148 

    assert args.ntrain <= ntrain_total, f"Only {args.ntrain_total} data "\
        f"available in {args.data} dataset, but needs {args.ntrain} training data."
    assert args.ntest <= ntest_total, f"Only {args.ntest_total} data "\
        f"available in {args.data} dataset, but needs {args.ntest} test data." 

    train_loader, _ = load_data(train_hdf5_file, args.ntrain, args.batch_size,
        only_input=True, return_stats=False)                       
    test_loader, test_stats = load_data(test_hdf5_file, args.ntest,    
        args.test_batch_size, only_input=False, return_stats=True) 
    y_test_variation = test_stats['y_variation']            
    print(f'Test output variation per channel: {y_test_variation}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                        weight_decay=args.weight_decay)        
    scheduler = OneCycleScheduler(lr_max=args.lr, div_factor=args.lr_div, 
                        pct_start=args.lr_pct)
    sobel_filter = SobelFilter(args.imsize, correct=True, device=device)

    n_out_pixels = test_loader.dataset[0][1].numel() 
    print(f'Number of out pixels per image: {n_out_pixels}') 

    logger = {}
    logger['loss_train'] = []
    logger['loss_test'] = []
    logger['r2_test'] = []
    logger['nrmse_test'] = []
    logger['nrmse_test_origin'] = []

    def test(epoch):
        model.eval()       
        loss_test = 0.
        relative_l2, err2 = [], []
        relative_l2_origin, err2_origin = [], []
        for batch_idx, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)   
            time_pre=time.time()
            output = model(input)          
            time_pre2=time.time()
            print(f'Finished predicting {args.test_batch_size} samples using: {(time_pre - time_pre2):.6f} secs')
            loss_pde1, loss_pde2 = constitutive_constraint(input, output, sobel_filter)
            loss_pde3 = continuity_constraint(output, sobel_filter)
            loss_pde = loss_pde1 + 10*loss_pde2 + loss_pde3
            loss_dirichlet, loss_neumann = boundary_condition(output)
            loss_boundary = loss_dirichlet + loss_neumann         
            loss = loss_pde + loss_boundary * args.weight_bound
            loss_test += loss.item()

            # sum over H, W --> (B, C)
            err2_sum_origin = torch.sum((output - target) ** 2, [-1, -2])
            relative_l2_origin.append(torch.sqrt(err2_sum_origin / (target ** 2).sum([-1, -2])))
            err2_origin.append(err2_sum_origin)                                 

            output1=output[:,[0,1]]
            output1[:,[1]]=torch.sqrt(output[:,[1]]**2+output[:,[2]]**2)
            target1=target[:,[0,1]]
            target1[:,[1]]=torch.sqrt(target[:,[1]]**2+target[:,[2]]**2)
            err2_sum = torch.sum((output1 - target1) ** 2, [-1, -2])
            relative_l2.append(torch.sqrt(err2_sum / (target1 ** 2).sum([-1, -2])))
            err2.append(err2_sum)

            # plot predictions
            if (epoch % args.plot_freq == 0 or epoch == args.epochs) and \
                batch_idx == len(test_loader) - 1:                                        
                n_samples = 512 if epoch == args.epochs else 2                          
                
                if (epoch == args.epochs):
                    for ii in range(args.test_batch_size):
                        print(f"writing test samples {ii}")
                        samples_input1=input.data.cpu()[ii].numpy()
                        samples_output1=output.data.cpu()[ii].numpy()
                        samples_target1=target.data.cpu()[ii].numpy()
                        np.savetxt(args.pred_dir+f"/input{ii}.txt", samples_input1[0])
                        np.savetxt(args.pred_dir+f"/output{ii}.txt", samples_output1[0])
                        np.savetxt(args.pred_dir+f"/target{ii}.txt", samples_target1[0])
                        np.savetxt(args.pred_dir+f"/target_fx{ii}.txt", samples_target1[1])
                        np.savetxt(args.pred_dir+f"/target_fy{ii}.txt", samples_target1[2])
                        np.savetxt(args.pred_dir+f"/output_fx{ii}.txt", samples_output1[1])
                        np.savetxt(args.pred_dir+f"/output_fy{ii}.txt", samples_output1[2])
                        del samples_input1
                        del samples_output1
                        del samples_target1

                idx = torch.randperm(input.size(0))[:n_samples]
                samples_output = output.data.cpu()[idx].numpy()
                samples_target = target.data.cpu()[idx].numpy()
                samples_output1 = output1.data.cpu()[idx].numpy()
                samples_target1 = target1.data.cpu()[idx].numpy()
                for i in range(n_samples):
                    print('epoch {}: plotting prediction {}'.format(epoch, i))
                    plot_prediction_det(args.pred_dir, samples_target[i], samples_output[i], epoch, i, plot_fn=args.plot_fn)          
                    plot_prediction_det2(args.pred_dir, samples_target1[i], samples_output1[i], epoch, i, plot_fn=args.plot_fn)         

        loss_test /= (batch_idx + 1)
        relative_l2 = to_numpy(torch.cat(relative_l2, 0).mean(0))
        relative_l2_origin = to_numpy(torch.cat(relative_l2_origin, 0).mean(0))
        r2_score = 1 - to_numpy(torch.cat(err2_origin, 0).sum(0)) / y_test_variation              
        print(f"Epoch: {epoch}, test r2-score:  {r2_score}")
        print(f"Epoch: {epoch}, test relative-l2:  {relative_l2}")
        print(f'Epoch {epoch}: test loss: {loss_train:.6f}, loss_pde: {loss_pde.item():.6f}, '\
                f'dirichlet {loss_dirichlet:.6f}, nuemann {loss_neumann.item():.6f}')            

        if epoch % args.log_freq == 0:
            logger['loss_test'].append(loss_test)
            logger['r2_test'].append(r2_score)
            logger['nrmse_test'].append(relative_l2)            
            logger['nrmse_test_origin'].append(relative_l2_origin)

    print('Start training...................................................')
    start_epoch = 1 if args.ckpt_epoch is None else args.ckpt_epoch + 1  
    tic = time.time()
    # step = 0
    total_steps = args.epochs * len(train_loader)   
    print(f'total steps: {total_steps}')
    for epoch in range(start_epoch, args.epochs + 1):
        time_train=time.time()
        model.train()                          
        loss_train, mse = 0., 0.
        for batch_idx, (input, ) in enumerate(train_loader, start=1):           
            input = input.to(device)
            model.zero_grad()
            output = model(input)
            loss_pde1, loss_pde2 = constitutive_constraint(input, output, sobel_filter)
            loss_pde3 = continuity_constraint(output, sobel_filter)
            loss_pde = loss_pde1 + 10*loss_pde2 + loss_pde3
            loss_dirichlet, loss_neumann = boundary_condition(output)            
            loss_boundary = loss_dirichlet + loss_neumann                      
            loss = loss_pde + loss_boundary * args.weight_bound
            loss.backward()
            # lr scheduling
            step = (epoch - 1) * len(train_loader) + batch_idx
            pct = step / total_steps
            lr = scheduler.step(pct)
            n = epoch // 10
            lr = 1e-3 * 0.9**n
            adjust_learning_rate(optimizer, lr)
            optimizer.step()
            loss_train += loss.item()

        loss_train /= batch_idx
        time_train2=time.time()
        print(f'Epoch {epoch}, lr {lr:.6f}')
        print(f'Epoch {epoch}: training loss: {loss_train:.6f}, pde: {loss_pde:.6f}, '\
            f'dirichlet {loss_dirichlet:.6f}, nuemann {loss_neumann:.6f}')
        print(f'Finished training epoch {epoch} using: {(time_train - time_train2) / 60:.2f} mins')
        if epoch % args.log_freq == 0:
            logger['loss_train'].append(loss_train)                                            
        if epoch % args.ckpt_freq == 0:
            torch.save(model.state_dict(), args.ckpt_dir + "/model_epoch{}.pth".format(epoch)) 
        
        with torch.no_grad():
            time_test=time.time()
            test(epoch)
            time_test2=time.time()
            print(f'Finished test epoch {epoch} using: {(time_test - time_test2) / 60:.2f} mins')     

    tic2 = time.time()
    print(f'Finished training {args.epochs} epochs with {args.ntrain} data ' \
        f'using {(tic2 - tic) / 60:.2f} mins')                                 
    metrics = ['loss_train', 'loss_test', 'nrmse_test','nrmse_test_origin', 'r2_test']             
    save_stats(args.train_dir, logger, *metrics)                              
    args.training_time = tic2 - tic
    args.n_params, args.n_layers = model.model_size
    with open(args.run_dir + "/args.txt", 'w') as args_file:
        json.dump(vars(args), args_file, indent=4)                  
