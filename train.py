import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
#from torch.tensorboard import SummaryWriter
#writer = SummaryWriter("runs/pothole")
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

import gc

gc.collect()

torch.cuda.empty_cache()
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--model_path', help='Path where values post each iterative are stored.')
	
    parser = parser.parse_args(args)
    dir=parser.model_path
    print(dir)
	
    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
## sample oof Datase_train is a dictionary {"img":img, "annot":annot}
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
            #print(dataset_val)
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    #sampler = AspectRatioBasedSampler(dataset_train, batch_size=4, drop_last=False)
    #print(dataset_train.num_classes)
    dataloader_train = DataLoader(dataset_train, num_workers=1, collate_fn=collater, batch_size=1, shuffle=True)
    #annotation = np.zeros((1, 5))
    print(" len ",len(dataloader_train))
    #print(dataloader_train[0],"  2  ",dataloader_train[1])
    for x,annotation in enumerate(dataloader_train):
        print(" X  ", x)
        print("y: ", annotation['annot'])
        break
  
    if dataset_val is not None:
        #sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=4, drop_last=False)
        #print("2",dataset_val)
        dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_size=1, shuffle=True)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    #to load a pretrained model
    #retinanet=torch.load('csv_retinanet_188_pothole.pt')  
	model_children = list(retinanet.children())
	
	
    use_gpu = True
    #retinanet.load(torch.load('csv_retinanet_195.pt'))
    #model=models.retinanet()
    print(retinanet)
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    
    retinanet.register_forward_hook(get_activation('layer0'))
    #retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))
    eel = []
    ecl = []
    erl = []
    for epoch_num in range(parser.epochs):

        retinanet.train()
        #retinanet.module.freeze_bn()

        epoch_loss = []
        cl = []
        rl = []
        
       

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()
                cl.append(float(classification_loss))
                rl.append(float(regression_loss))
                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print("exception : +",e)
                continue

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')
            print('Evaluating dataset',dataset_val )
            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, 'pothole_detection/model/{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))
        ecl.append(np.mean(cl))  
		
        erl.append(np.mean(rl)) 
        eel.append(np.mean(epoch_loss))
		
        del cl
        del rl
        del epoch_loss
		
    retinanet.eval()
    plt.matshow(activations['layer0'])





    plt.matshow(weights)
    plt.show()
    #output = np.asarray([eel,ecl, erl])	
    output = np.column_stack(((np.array(eel)).flatten(),(np.array(ecl)).flatten(), (np.array(erl)).flatten()))		
    np.savetxt('output_metric.csv',output,delimiter=',')
    torch.save(retinanet, 'model_final_newds.pt')


if __name__ == '__main__':
    main()
