# Learning to Compose Hypercolumns for Visual Correspondence
This is the implementation of the paper "Learning to Compose Hypercolumns for Visual Correspondence" by J. Min, J. Lee, J. Ponce and M. Cho. Implemented on Python 3.7 and PyTorch 1.0.1.

![](https://juhongm999.github.io/pic/dhpf.png)

For more information, check out project [[website](http://cvlab.postech.ac.kr/research/DHPF/)] and the paper on [[arXiv](https://arxiv.org/abs/2007.10587)].


## Requirements

- Python 3.7
- PyTorch 1.0.1
- tensorboard
- scipy
- pandas
- requests
- scikit-image

Conda environment settings:
```bash
conda create -n dhpf python=3.7
conda activate dhpf

conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch
pip install tensorboardX
conda install -c anaconda scipy
conda install -c anaconda pandas
conda install -c anaconda requests
conda install -c anaconda scikit-image
```

## Training

Training DHPF with <b>strong supervision</b> (keypoint annotations) on PF-PASCAL and SPair-71k</br>
(reproducing strongly-supervised results in Tab. 1 and 2): 
```bash
python train.py --supervision strong \
                --lr 0.03 \
                --bsz 8 \
                --niter 100 \
                --selection 0.5 \ 
                --benchmark pfpascal \ 
                --backbone {resnet50, resnet101}

python train.py --supervision strong \
                --lr 0.03 \
                --bsz 8 \
                --niter 5 \
                --selection 0.5 \ 
                --benchmark spair \ 
                --backbone {resnet50, resnet101}
```
Training DHPF with <b>weak supervision</b> (image-level labels) on PF-PASCAL</br>
(reproducing weak-supervised results in Tab. 1):
```bash
python train.py --supervision weak \
                --lr 0.1 \
                --bsz 4 \
                --niter 30 \
                --selection 0.5 \  
                --benchmark pfpascal \
                --backbone {resnet50, resnet101}
```

## Testing

We provide trained models available on [[Google drive](https://drive.google.com/drive/folders/1aoKQlvHOb7vZIFK8pDJsQnC7SOyEjXVF?usp=sharing)].

PCK @ α<sub>img</sub>=0.1 on PF-PASCAL at different μ:
 
| Trained models<br>at differnt μ |  0.3 |  0.4 |  0.5 |  0.6 |  0.7 |  0.8 |  0.9 |   1  |
|:--------------------------------------------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|                    [weak (res50)](https://drive.google.com/drive/folders/1WykysKyy9PAsX-DpC5UuZILokCMToJWH?usp=sharing)                    | 77.3 |  79  |  79  | 79.3 | 79.6 | 80.7 | 81.1 | 80.7 |
|                    [weak (res101)](https://drive.google.com/drive/folders/1IjjoFgrIZzys2YDEGhLQrOg0bTG29-Pl?usp=sharing)                   | 80.3 | 81.2 | 82.1 | 80.1 | 81.7 | 80.9 | 81.3 | 81.3 |
|                   [strong (res50)](https://drive.google.com/drive/folders/1RC9EbVhk8QOjpF3NIO-tidIsKcY399S8?usp=sharing)                   | 87.7 | 89.1 | 88.9 | 88.5 | 89.4 | 89.1 |  89  | 89.5 |
|                   [strong (res101)](https://drive.google.com/drive/folders/1QDYOxqF-BsWKjKbwLKfbcfxaS5OHlbVT?usp=sharing)                  | 88.7 |  90  | 90.7 | 90.2 | 90.1 | 90.6 | 90.6 | 90.4 |

PCK @ α<sub>img</sub>=0.1 on SPair-71k at μ=0.5:

| Trained models<br>at μ=0.5 |  PCK |
|:---------------------------------------------:|:----:|
|                 [weak (res101)](https://drive.google.com/file/d/1uDfONwSiAzDsxW9wbhdlYKf8auqAVXoM/view?usp=sharing)                 | 27.7 |
|                [strong (res101)](https://drive.google.com/file/d/1DnsDhttMIImAcupdjuANowlgZqVSx_5E/view?usp=sharing)                | 37.3 |

Reproducing results in Tab. 1, 2 and 3:
```bash
python test.py --backbone {resnet50, resnet101} \
               --benchmark {pfpascal, pfwillow, caltech, spair} \
               --load "path_to_trained_model"
```
    
   
## BibTeX
If you use this code for your research, please consider citing:
````BibTeX
@InProceedings{min2020dhpf, 
   title={Learning to Compose Hypercolumns for Visual Correspondence},
   author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
   booktitle={ECCV},
   year={2020}
}
````
