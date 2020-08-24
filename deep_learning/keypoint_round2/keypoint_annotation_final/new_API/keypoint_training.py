import os, random, time, copy
import numpy
import os.path as path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torchvision
from torchvision import datasets, models, transforms
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from zplib.curve import interpolate
from zplib.image import pyramid
import keypoint_annotation_model
from elegant import worm_spline
#since all the worms will be in the same orientation/worm pixels, hardcode in a worm_frame_mask
def to_tck(widths):
    x = numpy.linspace(0, 1, len(widths))
    smoothing = 0.0625 * len(widths)
    return interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

WIDTH_TRENDS = pickle.load(open('/home/nicolette/.conda/envs/nicolette/lib/python3.7/site-packages/elegant/width_data/width_trends.pickle', 'rb'))
AVG_WIDTHS = numpy.array([numpy.interp(5, WIDTH_TRENDS['ages'], wt) for wt in WIDTH_TRENDS['width_trends']])
AVG_WIDTHS_TCK = to_tck(AVG_WIDTHS)

class LossofRegmentation(nn.Module):
    def __init__(self, downscale=2, scale=(0,1,2,3), image_shape=(960,512)):
        super(LossofRegmentation, self).__init__()
        self.scale = scale
        self.reglLoss = nn.L1Loss(reduction='sum')
        #self.segLoss = nn.BCELoss(reduction='sum')
        self.downscale = downscale
        image_size = (int(image_shape[0]/downscale), int(image_shape[1]/downscale))
        widths_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])
        mask = worm_spline.worm_frame_mask(widths_tck, image_size) #make worm mask for training
        self.mask = mask

    def forward(self, Keypoint0, Output):
            
        K0loss = 0  
        ##image1 mask image2 output
        for i in self.scale: 
            s = 2**i      
            N,C,H,W = Keypoint0[i].size()
            scaled_mask = pyramid.pyr_down(self.mask, downscale=s)
            m = numpy.array([[scaled_mask]*C]*N) #get mask into the same dimension as keypoint should be (N, 1, H, W)
            tensor_mask = torch.tensor(m) #make the mask into a tensor
            l = self.reglLoss(Output[('Keypoint0',i)][tensor_mask>0], Keypoint0[i][tensor_mask>0])/(N*C*H*W)
            print('Loss: {}, scale: {}'.format(l, i))
            K0loss += l

        return K0loss


def training_wrapper(dataloaders, dataset_sizes, loss_1_to_2, base_lr = 0.0001 ,scale=[0,1,2,3], 
            start_epo = 0, total_epoch_nums=25, work_dir='./', device='cpu'):

    log_filename = os.path.join(work_dir,'train.log')    
    for i, keypoint in enumerate(['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        since = time.time()
        curr_time = datetime.now()
        print('------------------------{} Training {} ------------------------'.format(curr_time, keypoint))
        print('base_lr: {}\t scale: {}\t start_epo: {}\t total_epoch_nums: {}\t device: {}\t work_dir: {}\t'.format(
            base_lr, scale, start_epo, total_epoch_nums, device, work_dir))
        print('dataloader sizes: {}:{}\t {}:{}\t'.format('train', dataset_sizes['train'], 'val', dataset_sizes['val']))
        fn = open(log_filename, 'a')
        fn.write('------------------------{} Training {} ------------------------\n'.format(curr_time, keypoint))
        #fn.write('base_lr: {}\t scale: {}\t start_epo: {}\t total_epoch_nums: {}\t device: {}\t work_dir: {}\n'.format(
         #   base_lr, scale, start_epo, total_epoch_nums, device, work_dir))
        fn.write('dataloader sizes: {}:{}\t {}:{}\n'.format('train', dataset_sizes['train'], 'val', dataset_sizes['val']))
        fn.close()
        #initialize model
        initModel = keypoint_annotation_model.WormRegModel(34, scale, pretrained=True)
        initModel.to(device)
        #define loss function
        loss_1_to_2 = loss_1_to_2
        optimizer = torch.optim.Adam([{'params': initModel.parameters()}], lr=base_lr)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_epoch_nums/10), gamma=0.5)

        #train the model
        model_ft = train_reg(initModel, dataloaders, dataset_sizes, loss_1_to_2, optimizer, exp_lr_scheduler, i, keypoint, 
            start_epo=0, num_epochs=total_epoch_nums, work_dir=work_dir, device=device)

        print('----------------------------------------------------------------------------')
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        fn = open(log_filename,'a')
        fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        fn.close()

def train_reg(model, dataloaders, dataset_sizes, loss_1_to_2, optimizer, scheduler, 
                keypoint_idx, keypoint_name, start_epo = 0, num_epochs=25, work_dir='./', device='cpu'):
    
    save_dir = os.path.join(work_dir, keypoint_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    log_filename = os.path.join(save_dir,'train.log')    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(start_epo ,num_epochs):        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(phase)
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else: model.eval()   # Set model to evaluate mode

            #running_loss, running_lossk0, running_lossk1, running_lossk2, running_lossk3 = 0.0, 0.0, 0.0, 0.0, 0.0
            running_loss, running_lossk0, running_acc = 0.0, 0.0, 0.0

            # Iterate over data.
            img, keypoint_maps, out = None, None, None
            iterCount,sampleCount = 0, 0
            for sample in dataloaders[phase]:
                img, keypoint_maps = sample 
                keypoint0 = keypoint_maps[keypoint_idx] 
                img = img.to(device)
                for i in range(len(keypoint0)):
                    keypoint0[i] = keypoint0[i].to(device)
                    #keypoint1[i] = keypoint1[i].to(device)
                    #keypoint2[i] = keypoint2[i].to(device)
                    #keypoint3[i] = keypoint3[i].to(device)
                    
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train                
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  # backward + optimize only if in training phase
                        model.train()                        
                        output = model(img)                          
                        k0loss = loss_1_to_2(keypoint0, output)
                        loss = k0loss
                        loss.backward()
                        optimizer.step()
                        acc = accuracy([keypoint0[0]], output)
                        
                    else: 
                        model.eval()                        
                        output = model(img)                          
                        k0loss = loss_1_to_2(keypoint0, output)
                        loss = k0loss
                        acc = accuracy([keypoint0[0]], output)

                # statistics  
                iterCount += 1
                sampleCount += img.size(0)                                
                running_loss += loss.item() * img.size(0) 
                running_lossk0 += k0loss.item() * img.size(0)
                running_acc += acc
                #running_lossk1 += k1loss.item() * img.size(0)
                #running_lossk2 += k2loss.item() * img.size(0) 
                #running_lossk3 += k3loss.item() * img.size(0)
                accprint2screen_avgLoss = running_acc/sampleCount
                k0print2screen_avgLoss = running_lossk0/sampleCount
                #k1print2screen_avgLoss = running_lossk1/sampleCount
                #k2print2screen_avgLoss = running_lossk2/sampleCount
                #k3print2screen_avgLoss = running_lossk3/sampleCount
                print2screen_avgLoss = running_loss/sampleCount
                                               
                if iterCount%50==0:
                    print('\t{}/{} loss: {:.4f} \t k0loss: {:.4f}\t acc: {:.4f}'.format(iterCount, len(dataloaders[phase]), print2screen_avgLoss, k0print2screen_avgLoss, accprint2screen_avgLoss))
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss: {:.4f} \t k0loss: {:.4f}\t acc: {:.4f}\n'.format(iterCount, len(dataloaders[phase]), print2screen_avgLoss, k0print2screen_avgLoss, accprint2screen_avgLoss))
                    fn.close()
  
            epoch_loss = running_loss / dataset_sizes[phase]
            k0epoch_loss = running_lossk0 / dataset_sizes[phase]
            accepoch_loss = running_acc / dataset_sizes[phase]
            #k1epoch_loss = running_lossk1 / dataset_sizes[phase]
            #k2epoch_loss = running_lossk2 / dataset_sizes[phase]
            #k3epoch_loss = running_lossk3 / dataset_sizes[phase]
                                
            print('\tloss: {:.6f} \tk0loss: {:.6f}\t acc: {:.6f}'.format(epoch_loss, k0epoch_loss, accepoch_loss))
            fn = open(log_filename,'a')
            fn.write('\tloss: {:.6f} \tk0loss: {:.6f} acc: {:.6f}\n'.format(epoch_loss, k0epoch_loss, accepoch_loss))
            fn.close()
            
            keypoint_maps = [keypoint0[0]]

            plot_output(img, keypoint_maps, output, epoch, phase, save_dir)
                
            # deep copy the model
            cur_model_wts = copy.deepcopy(model.state_dict())
            path_to_save_paramOnly = os.path.join(save_dir, 'epoch-{}.paramOnly'.format(epoch+1))
            torch.save(cur_model_wts, path_to_save_paramOnly)
            
            if phase=='val' and epoch_loss<best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                numpy.save('best', best_loss)
                
                path_to_save_paramOnly = os.path.join(save_dir, 'bestValModel.paramOnly')
                torch.save(best_model_wts, path_to_save_paramOnly)
                #path_to_save_wholeModel = os.path.join(save_dir, 'bestValModel.wholeModel')
                #torch.save(model, path_to_save_wholeModel)
                
                file_to_note_bestModel = os.path.join(save_dir,'note_bestModel.log')
                fn = open(file_to_note_bestModel,'a')
                fn.write('The best model is achieved at epoch-{}: loss{:.6f}.\n'.format(epoch+1,best_loss))
                fn.close()
                
        scheduler.step()       
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
   
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def accuracy(keypoint_maps, out):
    acc = 0
    N,C,H,W = keypoint_maps[0].size()
    s = int(960/H)#get the mask
    widths_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/s, AVG_WIDTHS_TCK[2])
    mask = worm_spline.worm_frame_mask(widths_tck, (H, W)) #make worm mask
    mask = mask>0
    print(mask.shape)
    #mask = numpy.array([[mask]*C]*N) #get mask into the same dimension as keypoint should be (N, 1, H, W)

    for sampleIndex in range(len(keypoint_maps[0])):
        kp_map = keypoint_maps[0][sampleIndex].cpu().numpy()
        gt = kp_map[0]
        gt[~mask] = -1 #since we don't care about things outside of the worm pixels, set everything outside to -1
        gt_kp = numpy.unravel_index(numpy.argmax(gt), gt.shape)
        #gt_kp = numpy.where(gt == numpy.max(gt[mask]))

        out_kp_map = out[('Keypoint0',0)][sampleIndex].cpu().detach().numpy()
        pred = out_kp_map[0]
        pred[~mask] = -1 #since we don't care about things outside of the worm pixels, set everything outside to -1
        #out_kp = numpy.where(pred == numpy.max(pred[mask]))
        out_kp = numpy.unravel_index(numpy.argmax(pred), pred.shape)

        #dist = numpy.sqrt((gt_kp[0][0]-out_kp[0][0])**2 + (gt_kp[1][0]-out_kp[1][0])**2)
        dist = numpy.sqrt((gt_kp[0]-out_kp[0])**2 + (gt_kp[1]-out_kp[1])**2)
        print("GT: {}, Out: {}, dist: {:.0f} ".format(gt_kp, out_kp, dist))
        acc += dist
    print("avg acc: ", acc/N)
    return acc


def plot_output(imgList, keypoint_maps, out, epoch, phase, save_dir='./'):
    figWinNumHeight, figWinNumWidth, subwinCount = 4, 4, 1
    plt.figure(figsize=(22,20), dpi=88, facecolor='w', edgecolor='k') # figsize -- inch-by-inch
    plt.clf()
    print(imgList.min())
    print(imgList.max())
    acc = 0
    N,C,H,W = keypoint_maps[0].size()
    print(N,C,H,W)
    s = int(960/H)#get the mask
    widths_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/s, AVG_WIDTHS_TCK[2])
    mask = worm_spline.worm_frame_mask(widths_tck, (H, W)) #make worm mask
    mask = mask>0


    for sampleIndex in range(min(4, len(imgList))):
        # visualize image
        plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
        subwinCount += 1
        image = imgList[sampleIndex].cpu().numpy()#.squeeze().transpose((1,2,0))      
        plt.imshow(image[0], cmap='gray')    
        plt.axis('off')
        plt.title('Image of worm')
        
        #keypoint 0
        plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
        subwinCount += 1
        kp_map = keypoint_maps[0][sampleIndex].cpu().numpy()#.squeeze().transpose((1,2,0))
        plt.imshow(kp_map[0], cmap='jet')
        plt.axis('on')
        plt.colorbar()
        plt.title('Keypoint '+str(0)+" GT")
        
        plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
        subwinCount += 1
        kp_map = out[('Keypoint0',0)][sampleIndex].cpu().detach().numpy()#.squeeze().transpose((1,2,0))
        plt.imshow(kp_map[0], cmap='jet')
        plt.axis('on')
        plt.colorbar()
        plt.title('Keypoint '+str(0))
        
        plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
        subwinCount += 1
        kp_map = out[('Keypoint0',0)][sampleIndex].cpu().detach().numpy()
        per90 = numpy.percentile(kp_map[0], 95)
        kp_map[0][~mask] = 0
        
        plt.imshow((kp_map[0]>50).astype(numpy.float32)*1, cmap='jet')
        plt.axis('on')
        plt.colorbar()
        plt.title('Keypoint '+str(0))

        """#Keypoint 1
                        
                                subwinCount+=1
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = keypoint_maps[1][sampleIndex].cpu().numpy()#.squeeze().transpose((1,2,0))
                                plt.imshow(kp_map[0], cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(1)+" GT")
                                
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = out[('Keypoint1',0)][sampleIndex].cpu().detach().numpy()#.squeeze().transpose((1,2,0))
                                plt.imshow(kp_map[0], cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(1))
                                
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = out[('Keypoint1',0)][sampleIndex].cpu().detach().numpy()
                                per90 = numpy.percentile(kp_map[0], 95)
                                
                                plt.imshow((kp_map[0]>per90).astype(numpy.float32)*1, cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(1))
                        
                                #Keypoint 2                                       
                                subwinCount+=1
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = keypoint_maps[2][sampleIndex].cpu().numpy()#.squeeze().transpose((1,2,0))
                                plt.imshow(kp_map[0], cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(2)+" GT")
                                
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = out[('Keypoint2',0)][sampleIndex].cpu().detach().numpy()#.squeeze().transpose((1,2,0))
                                plt.imshow(kp_map[0], cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(2))
                                
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = out[('Keypoint2',0)][sampleIndex].cpu().detach().numpy()
                                per90 = numpy.percentile(kp_map[0], 95)
                                
                                plt.imshow((kp_map[0]>per90).astype(numpy.float32)*1, cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(2))
                        
                                #keypoint 3
                                subwinCount+=1
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = keypoint_maps[3][sampleIndex].cpu().numpy()#.squeeze().transpose((1,2,0))
                                plt.imshow(kp_map[0], cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(3)+" GT")
                                
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = out[('Keypoint3',0)][sampleIndex].cpu().detach().numpy()#.squeeze().transpose((1,2,0))
                                plt.imshow(kp_map[0], cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(3))
                                
                                plt.subplot(figWinNumHeight,figWinNumWidth,subwinCount)
                                subwinCount += 1
                                kp_map = out[('Keypoint3',0)][sampleIndex].cpu().detach().numpy()
                                per90 = numpy.percentile(kp_map[0], 95)
                                
                                plt.imshow((kp_map[0]>per90).astype(numpy.float32)*1, cmap='jet')
                                plt.axis('on')
                                plt.colorbar()
                                plt.title('Keypoint '+str(3))"""
                        


    save_path = os.path.join(save_dir, ('epoch '+str(epoch)+' output '+phase+'.png'))
    plt.savefig(save_path)
    plt.close()