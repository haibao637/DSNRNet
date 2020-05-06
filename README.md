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
### x4 super resolution
|-|-|
|--|--|
|1/4 bicubic下采样|![](./docs/lr.png)
|x4 bicubic 上采样|![](./docs/bicubic.png)
|x4 PSNRNet|![](./docs/sr.png)
|原图|![](./docs/hr.png)

