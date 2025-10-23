import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.trainInstanceMonuSeg import DualDecoderUNet, post_process_remove_fps
from pathlib import Path
import json
import tifffile
from utils.utils import dilate_instances_no_overlap
from utils.utils_inference import (validate_with_augmentations_and_ensembling, add_pad ,remove_pad,
                             Dataset_test)
from torch.utils.data import DataLoader
from skimage.segmentation import watershed
import time
from skimage.morphology import remove_small_objects
import scipy

def load_data(data_path = ''):

    X_all = []
    for image in data_path:
        # print(image)
        X_all.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB).astype('float32'))
    try:
        X_all = np.array(X_all)
        X_all = X_all
    except Exception as e:
        print(e)
    return X_all



def AM_postProccess(preds_dis = None, pred_binary = None):
    markers_pure_dis = scipy.ndimage.maximum_filter(preds_dis, footprint=np.ones((10, 10)))

    markers_pure_dis = markers_pure_dis == preds_dis

    nSeedsCount, markers_pure_dis = cv2.connectedComponents(np.uint8(markers_pure_dis))


    pred_inst = watershed(1 - pred_binary, markers_pure_dis, mask=pred_binary, connectivity=2)
    return pred_inst

def inst_fast(pred_cent = None,th = 0.5):
    cent_certain = np.array( pred_cent > th, dtype=np.uint8)
    cent_certain = cv2.connectedComponents(np.array(1 - cent_certain, dtype=np.uint8))
    cent_certain = cent_certain[1]
    cent_certain = remove_small_objects(cent_certain, min_size=20)

    return cent_certain

def add_gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def inference(data_path,res,fold = None, save_im = False, fast = False,out_pth = '',
              gts = None, in_channels = 3,add_noise = False, normal_dist = False,
              debug = False):
    # out_dir="/output6/"
    # pred_path = '/home/ntorbati/STORAGE/PumaDataset/preds6/'#
    val_images = load_data(data_path=data_path)
    # val_images = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims_panopt.npy')
    device2 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    val_set = Dataset_test(val_images,
                                n_class1=6,
                                size1=(1024,1024),
                                     paths=res,
                                     device1=device2)
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    dataloader = DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args)
    if fold is None:
        n_folds = range(5)
    else:
        n_folds = [fold]

    model = DualDecoderUNet(encoder_name='resnet34',
                            encoder_weights=None,
                            in_channels=in_channels,
                            classes_semantic=1)
    predicted_inst = []
    kk = 0
    with torch.no_grad():
        for image, pth in dataloader:
            if add_noise:
                blurred = np.array(image[0].permute(1,2,0))
                blurred = add_gaussian_noise(blurred*255, mean=10, std=100)/255


                blurred = cv2.GaussianBlur(blurred, (50, 50), 0)
                image += torch.Tensor(blurred).permute(2,0,1).unsqueeze(0).float()
                image = image/2
            if in_channels > 3:
                nuc_path = pth[0].replace('.tif','.npy')
                tis_im = torch.tensor(tifffile.imread(pth[0])*50, dtype=torch.float32)
                tis_im /= 255
                nuc_im = torch.tensor(np.load(nuc_path)*25, dtype=torch.float32)


                nuc_im /= 255
            if in_channels > 3:
                image = torch.concatenate([image,tis_im.unsqueeze(0).unsqueeze(0),nuc_im.unsqueeze(0).unsqueeze(0)],dim=1)
            image = image.to(device=device2, dtype=torch.float32, memory_format=torch.channels_last)
            weights_list = []



            model_weight_path = 'inst'
            os.getcwd()
            for folds in n_folds:#range(n_folds):
                dir_checkpoint = Path(
                    'Model_Weights/'+ model_weight_path + str(
                        folds) + '/')

                PATH = os.path.join(os.getcwd(),str(dir_checkpoint / '{}_l2_l1_8_batch_dual_decoder_unet_monuseg1.pth'.format(folds)))
                weights_list.append(PATH)
            pred_inst_binary = validate_with_augmentations_and_ensembling(model=model,
                                                               image_tensor=image,
                                                               device=device2,
                                                               weights_list=weights_list,
                                                              regression= True,
                                                               )
            pred_inst_binary,pad = add_pad(pred_inst_binary, pad_siz=(20,20))
            pred_center = pred_inst_binary[0,:,:].cpu().numpy()
            # b_th = 0.5


            if out_pth != '':
                binary_pth = '/home/ntorbati/STORAGE/MoNuSAC/Test_GT/predictions6/'
                pred_binary = np.load(nuc_path.replace(out_pth,binary_pth))
                # plt.imshow(pred_binary)
                # plt.show()
                # ppp = np.copy(pred_binary)
                pred_binary,pad = add_pad(torch.tensor(pred_binary).unsqueeze(0), pad_siz=(20,20))
                pred_binary = np.array(pred_binary[0].cpu().numpy()>b_th,dtype=np.uint8)
            else:
                pred_binary = pred_inst_binary[1, :, :]

                pred_binary = pred_binary.cpu().numpy()
                b_th = np.mean(pred_binary[pred_binary > 0.1])
                pred_binary = pred_binary > b_th
            start = time.time()
            th = np.mean(pred_center[pred_center<0.9]) - 0.5*np.std(pred_center[pred_center<0.9])
            # th = np.mean(pred_center[pred_center>0.9]) - 0.5*np.std(pred_center[pred_center>0.9])


            pred_binary = dilate_instances_no_overlap(pred_binary, dilate=True)

            if normal_dist:
                pred_inst = AM_postProccess(preds_dis=pred_center, pred_binary=pred_binary)
            elif fast:

                pred_inst = inst_fast(pred_cent=pred_center,th = th)
                pred_inst = watershed(1-pred_binary, pred_inst, mask=pred_binary, connectivity=2)
                # pred_inst = remove_small_objects(pred_inst, min_size=20)

            else:
                # pred_binary = dilate_instances_no_overlap(pred_binary,dilate=True,iterations=1)


                pred_inst = post_process_remove_fps(pred_int1=pred_binary,
                                                    pred_cent = pred_center,
                                                    inst_th=th)
                # pred_inst = dilate_instances_no_overlap(pred_inst, dilate=True)
            end = time.time()
            print(f"Elapsed time: {end - start:.6f} seconds")

            # pred_inst = remove_small_objects(pred_inst, min_size=50)

            pred_inst = remove_pad(torch.tensor(pred_inst).unsqueeze(0).unsqueeze(0), pad)
            pred_inst = pred_inst[0,0].cpu().numpy()
            # pred_inst = dilate_instances_no_overlap(pred_inst,dilate=True)

            predicted_inst.append(pred_inst)
            pred_center = remove_pad(torch.tensor(pred_center).unsqueeze(0).unsqueeze(0), pad)
            pred_center = pred_center[0,0].cpu().numpy()

            # save JSON file
            if save_im:
                np.save(pth[0].replace('.tif','_inst.npy'),pred_inst)
                np.save(pth[0].replace('.tif','_center.npy'),pred_center)
            # json_filename = os.path.join('/output/melanoma-10-class-nuclei-segmentation.json')

            if debug:
                im = np.transpose(image[0].cpu().numpy(), (1, 2, 0))[:,:,0:3]
                plt.subplot(1, 3, 1)
                plt.imshow(im)

                pm = pred_inst
                plt.subplot(1, 3, 2)
                plt.imshow(pm)

                plt.subplot(1, 3, 3)
                plt.imshow(gts[kk])
                plt.show()
            kk+=1

    predicted_inst = np.stack(predicted_inst)
    return predicted_inst

if '__main__' == __name__:
    # save_segformer_config(6)
    path = '/input/images/melanoma-wsi/'
    out_path = '/output/images/melanoma-tissue-mask-segmentation/'
    #
    # path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/test/'
    # out_path = '/home/ntorbati/PycharmProjects/PUMA-challenge-baseline-track2/output/images/melanoma-tissue-mask-segmentation/'


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # all_tissue_data = np.sort([path + image for image in os.listdir(path)])
    images = [path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    res = [out_path + f for f in os.listdir(path) if f.endswith('.tif') and not f.endswith('_context.tif')]
    run(images, res)