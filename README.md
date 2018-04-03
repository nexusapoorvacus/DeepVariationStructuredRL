# DeepVariationStructuredRL
This repository contains a PyTorch implementation of the [Deep Variation-structured Reinforcement Learning for Visual Relationship and Attribute Detection](https://arxiv.org/abs/1703.03054) paper by Liang et. al [5].

## Setup
We will be using the [Visual Genome Dataset](http://visualgenome.org) to train this network.

## Training
To begin training the network, run

`python main.py --train`

## Evaluation
To evalutate a pretrained model, run

`python main.py --evaluate`

Add the `--visualize <number>` flag to save <number> scene graph diagrams to `visualizations/`.
  
## Sample Results & Visualizations

## Poster
This project was originally done for a Reinforcement Learning class at Stanford University (CS234). The poster for this project can be found [here](https://docs.google.com/presentation/d/1DKUT8oT75fstDhadKfCuODS3zHz26Mn1Oj9NvumFZQM/edit?usp=sharing) and the final report can be found [here](https://drive.google.com/file/d/10y1mYCvm7Q6Y4HLyBAmX2neYFcGwUl9x/view?usp=sharing).

More setup/train/test instruction updates coming soon.


Citations:

[1] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. Scene graph generation by iterative message passing. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

[2] Newell, Alejandro, and Jia Deng. “Pixels to Graphs by Associative Embedding.” [1706.07365] Pixels to Graphs by Associative Embedding, 22 June 2017, arxiv.org/abs/1706.07365.

[3] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, Michael Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. 2016.


[4] Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q, \textit{Densely connected convolutional networks}, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017

[5] X. Liang, L. Lee, and E. P. Xing. Deep variation-structured reinforcement
learning for visual relationship and attribute detection. In
CVPR, 2017

[6] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards
real-time object detection with region proposal networks. In NIPS,
2015

[7] Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David,
Rusu, Andrei A., Veness, Joel, Bellemare, Marc G.,
Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K.,
Ostrovski, Georg, Petersen, Stig, Beattie, Charles, Sadik,Amir, Antonoglou, Ioannis, King, Helen, Kumaran,
Dharshan, Wierstra, Daan, Legg, Shane, and Hassabis,
Demis. Human-level control through deep reinforcement
learning. Nature, 518(7540):529–533, 02 2015

[8] Van Hasselt, Hado, Guez, Arthur, and Silver, David. Deep
reinforcement learning with double q-learning. arXiv
preprint arXiv:1509.06461, 2015.
