# Image2StyleGan
I have used images of two fashion models and preprocessed facial part locations, then worked to morph two faces to create a 
facial template for potential virtual/simulated characters. 

## Acknowledgement & References 

Took zaidbhat1234/Image2StyleGAN implementation of the paper "Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?"

## Results 

<img width="514" alt="Screen Shot 2022-08-23 at 9 07 14 PM" src="https://user-images.githubusercontent.com/53489568/186296017-314487d0-ac8c-4371-ae1e-c8fcdf6f0557.png">

## Room for improvements 

1. Denoise the background. 
2. Facial distortions when facial parts are covered. 

<img width="482" alt="Screen Shot 2022-08-23 at 9 14 01 PM" src="https://user-images.githubusercontent.com/53489568/186296024-68be1c33-0902-428b-b98c-4ecd328b1d64.png">

## Directions 

1. Utilize the segmentation model to filter out the irrelevant background noises. 
2. Educate the model to reconstruct facial parts in the style of the target image based on their positional and conceptual information.
3. Incoporate the concept of "beauty standard" into the encoder. Aim to morph two images in a manner that complements one another. 
