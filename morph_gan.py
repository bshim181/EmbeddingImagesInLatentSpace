from stylegan_layers import  G_mapping,G_synthesis
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision import models
from torchvision.utils import save_image
import numpy as np
from math import log10
import matplotlib.pyplot as plt
from tqdm import tqdm
from perceptual_model import VGG16_perceptual
from SSIM import SSIM
from mlxtend.image import extract_face_landmarks
import scipy.ndimage
import pdb
import PIL 



#Load the pre-trained model

def loss_function(syn_img, img, img_p, MSE_loss, upsample, perceptual):

  #UpSample synthesized image to match the input size of VGG-16 input.
  #Extract mid level features for real and synthesized image and find the MSE loss between them for perceptual loss.
  #Find MSE loss between the real and synthesized images of actual size
  syn_img_p = upsample(syn_img)
  syn0, syn1, syn2, syn3 = perceptual(syn_img_p)
  r0, r1, r2, r3 = perceptual(img_p)
  mse = MSE_loss(syn_img,img)

  per_loss = 0
  per_loss += MSE_loss(syn0,r0)
  per_loss += MSE_loss(syn1,r1)
  per_loss += MSE_loss(syn2,r2)
  per_loss += MSE_loss(syn3,r3)

  return mse, per_loss


def PSNR(mse, flag = 0):
  #flag = 0 if a single image is used and 1 if loss for a batch of images is to be calculated
  if flag == 0:
    psnr = 10 * log10(1 / mse.item())
  return psnr

def processImage(img, landmark, output_size=1024, transform_size=4096, enable_padding = True):
    lm = np.array(landmark)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

# Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2


    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

        # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    return img

import imageio

def extract_landmark(path):
    img = imageio.imread(path)
    landmarks = extract_face_landmarks(img)
    
    return landmarks



def embedding_function(image, g_mapping, g_synthesis):
  upsample = torch.nn.Upsample(scale_factor = 256/1024, mode = 'bilinear')
  img_p = image.clone()
  img_p = upsample(img_p)
  #Perceptual loss initialise object
  perceptual = VGG16_perceptual().to(device)

  #MSE loss object
  MSE_loss = nn.MSELoss(reduction="mean")
  #since the synthesis network expects 18 w vectors of size 1x512 thus we take latent vector of the same size
  latents = torch.zeros((1,18,512), requires_grad = True, device = device)
  #Optimizer to change latent code in each backward step
  optimizer = optim.Adam({latents},lr=0.01,betas=(0.9,0.999),eps=1e-8)


  #Loop to optimise latent vector to match the generated image to input image
  loss_ = []
  loss_psnr = []
  count = 0
  ssim_loss = SSIM()
  print("Start Optimizing Latent Space") 
  for e in tqdm(range(500)):
    optimizer.zero_grad()
    syn_img = g_synthesis(latents)
    syn_img = (syn_img+1.0)/2.0
    ssim_loss_syn = upsample(syn_img)
    ssim_loss_out = -ssim_loss(ssim_loss_syn, img_p)
    mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
    psnr = PSNR(mse, flag = 0)
    loss = per_loss + mse 
    loss.backward()
    optimizer.step()
    loss_np=loss.detach().cpu().numpy()
    loss_p=per_loss.detach().cpu().numpy()
    loss_m=mse.detach().cpu().numpy()
    loss_psnr.append(psnr)
    loss_.append(loss_np)
    if (e+1)%250==0 :
      print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e+1,loss_np,loss_m,loss_p,psnr))
      count = count + 1 
      save_image(syn_img.clamp(0,1),"./imgs/progress{}.png".format(count))
  return latents


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_path = './imbedimgs/Vittoria.jpeg'
    with open(img_path,"rb") as f:
        image1=Image.open(f)
        image1=image1.convert("RGB")
    landmark1 = extract_landmark(img_path)
    image1 = processImage(image1, landmark1)
    transform = transforms.Compose([transforms.ToTensor()])
    image1 = transform(image1)
    image1 = image1.unsqueeze(0)
    image1 = image1.to(device)
    print("image1 loaded")

#Read a sample image we want to find a latent vector for
    img_path = './imbedimgs/DariaS.jpeg'
    with open(img_path,"rb") as f:
        image2=Image.open(f)
        image2=image2.convert("RGB")
    landmark2 = extract_landmark(img_path)
    image2 = processImage(image2, landmark2)
    transform = transforms.Compose([transforms.ToTensor()])
    image2 = transform(image2)
    image2 = image2.unsqueeze(0)
    image2 = image2.to(device)
    print("image2 loaded")


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis(resolution=1024))
    ]))

#Load the pre-trained model
    g_all.load_state_dict(torch.load('./PretrainedModel/karras2019stylegan-ffhq-1024x1024.pt', map_location=device))
    g_all.eval()
    g_all.to(device)
    g_mapping, g_synthesis = g_all[0],g_all[1]

    latent1 = embedding_function(image1, g_mapping, g_synthesis)
    latent2 = embedding_function(image2, g_mapping, g_synthesis)


    #Morph 
    for i in tqdm(range(40)):
        a = (1/40)*i
        w = latent1 * (1-a) + latent2 * a
        syn_img = g_synthesis(w)
        syn_img = (syn_img+1.0)/2.0
        save_image(syn_img.clamp(0,1),"./imgs/Morphed{}.png".format(i))
