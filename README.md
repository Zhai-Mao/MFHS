<div align="center">
<h2>MFHS:Mutual Consistency Learning-based foundation model integrates Hypergraph Computation for Semi-supervised Medical Image Segmentation</h2>
</div>
<div align="center">
Xuejun Liu, Zhaichao Tang, Yonghao Wu*, Ruixiang Zhai, Xuanhe Dong,
Zikang Dua, Shujun Cao
</div>
### abstract

Medical image segmentation is crucial for clinical practice, but fully supervised deep learning methods are hindered by the high cost of expert annotations.
Semi-Supervised Learning (SSL) offers a compelling solution by leveraging abundant unlabeled data, yet existing SSL methods face two primary limitations: they often neglect the rich, generalizable knowledge from vision foundation models, and their reliance on pairwise relationships inadequately captures complex anatomical structures.
In this paper, we propose MFHS, a novel semi-supervised framework that synergizes a vision foundation model with hypergraph learning to address these challenges.
Our framework leverages a pre-trained SAM2 encoder to extract robust hierarchical features, which are then refined by a hypergraph neural network that explicitly models many-to-many high-order relationships among anatomical structures.
For semi-supervised training, we employ a multi-decoder architecture to generate high-quality pseudo-labels through a cross-consistency mechanism, further enhanced by an adversarial learning module.
Extensive experiments on the ACDC dataset demonstrate that MFHS outperforms competitive  methods across various label ratios (5\%, 10\%, and 20\%).
Specifically, it achieves superior boundary delineation with the best HD95 scores at 5\% and 10\% labeled data, while attaining the highest Dice scores of 88.28\% and 89.21\% at 10\% and 20\% ratios, respectively.
The results confirm that combining pre-trained knowledge from foundation models with explicit high-order relational modeling leads to more accurate and robust semi-supervised medical image segmentation.
