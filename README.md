# Deep Network for Image/Video detail sharpening and noise removing
## example
### remove facial noise,enhance eye's detail
![](./docs/enhance_1%20(2).jpg)
### smooth background,enhance body's detail
![](./docs/enhance_1%20(1).jpg)
### preserve the high detail information
![](./docs/enhance_1%20(3).jpg)
### remove facial noise
![](./docs/enhance_1%20(5).jpg)
### remove environmental noise and enhance detail
![](./docs/enhance_1%20(6).jpg)
### the performance for video
<!-- ![](./docs/1586347400466825-converted.mp4) -->
<video src="./docs/1586347400466825-converted.mp4"  controls preload></video>
### super resolution & detail enhancing & facial smoothing , all in one model ， video order [origianl,super+detail enhance + noise remove),bilinear downsample+upsample]
<video src="./docs/1586347390584509_convert.m4v" width="960" height="540" controls preload></video>
### x2 super resolution (original x2 vs super resolution vs x2 cubic upsample )
- image size:360x288 x2 upsample
- upsample time : 0.00294
- fps : 340
- gpu 2080Ti
- model size: 125 KB
- spnr : 22.02 (TecoGAN 21.83)
<video src="./docs/calendar.mp4" width="960" height="540" controls preload></video>


