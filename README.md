## # DA4NMT
A list of awesome papers on NMT domain adaptation, but not restricted in NMT.
# Table of Contents

* [Papers](#papers)
  * [Surveys](#surveys)
  * [Supervised Domain Adaptation](Supervised-Domain-Adaptation-for-NMT)
    * [Instance-based Methods](#Instance-based-Methods)
      * [Data Selection](#Data-Selection)
      * [Data Sampling](#Data-Sampling)
      * [Instance Weighting](#Instance-Weighting)
    * [Parameter-based Methods](#Parameter-based-Methods)
      * [Fine-tuning Methods](#fine-tuning-Methods) 
      * [Regularized-based Methods](Regularized-based-Methods)
    * [Model-based Methods](#Model-based-Methods)
      * [Embedding methods](#embedding-methods)
      * [Adversarial methods](#adversarial-methods)
  * [Semi-supervised Domain Adaptation](#Semi-supervised-Domain-Adaptation-for-NMT)
  * [Unsupervised Domain Adaptation](#Unsupervised-Domain-Adaptation-for-NMT)
  * [Multi-source Domain Adaptation](#Multi-source-Domain-Adaptation-for-NMT)
  * [Mix-domain Domain Adaptation](#Mix-domain-Domain-Adaptation)
* [Datasets](#datasets)

# Papers

Papers are ordered by theme and inside each theme by general or NMT and then by the publication date (submission date for arXiv papers).

## Surveys

* [A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf) (2009)
    - 迁移学习必读survey，涵盖了早期的领域适应方法。
* [Deep Visual Domain Adaptation: A Survey](https://arxiv.org/abs/1802.03601) (CVPR2018)
    - 视觉领域适应的survey，内容很全，思路清晰易懂，推荐！
* [A Survey of Domain Adaptation for Neural Machine Translation](https://www.aclweb.org/anthology/C18-1111) (COLING2018)
    - NMT领域适应的survey，内容丰富。

## Supervised Domain Adaptation for NMT

Utilize in-domain/target domain parallel data.

### Instance-based Methods

#### Data Selection
* [Intelligent selection of language model training data]() (ACL2010)
* [Domain adaptation via pseudo in-domain data selection]() (EMNLP2011)
* [Adaptation data selection using neural language models: Experiments in machine translation]() (ACL2013)
* [Sentence embedding for neural machine translation domain adaptation]() (ACL2017)
    - 根据源域样本与目标域的相似性选择源域样本加入到目标域训练集中
* [Sentence embedding for neural machine translation domain adaptation]() (EMNLP2017)
    - dynamic data selection，在训练过程中改变可选择的子集，他们发现根据与目标域的相似性逐渐减少训练数据表现最好
 
#### Data Sampling
* [An empirical comparison of simple domain adaptation methods for neural machine translation]() (arxiv2017)
    - 对源域样本进行过采样
* [Sentence embedding for neural machine translation domain adaptation]() (ACL2017)
    - 根据样本embedding的相似性进行样本采样

#### Instance Weighting
* [Instance weighting for neural machine translation domain adaptation]() (EMNLP2017) 
    - 根据源域和目标域的LM的cross-entropy得到样本的权重，引入目标函数中
* [Cost weighting for neural machine translation domain adaptation]() (2017)
    - 用domain classifier的输出概率作为领域权重
* [Sentence selection and weighting for neural machine translation domain adaptation]() (TASLP2018)
    - 数据选择和加权的联合框架

### Parameter-based Methods

#### Fine-tuning Methods
##### General
* [Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1805.08974) (2018)
* [Using Pre-Training Can Improve Model Robustness and Uncertainty](https://arxiv.org/abs/1901.09960) (2019)
##### NMT
* [Stanford neural machine translation systems for spoken language domains](https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf) (2015)
    - 先在源域训练，之后在目标域finetune
* [Fast domain adaptation for neural machine translation]() (2016)
    - finetune后的模型与原模型集成
* **Mixed Finetuing**: [An empirical comparison of domain adaptation
methods for neural machine translation]() (ACL2017)
    - 用源域和目标域数据一起finetune (引入领域标记)

#### Regularized-based Methods
##### General
* **CORAL**: [Return of Frustratingly Easy Domain Adaptation](https://arxiv.org/abs/1511.05547) (2015)
* **Deep CORAL**: [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/abs/1607.01719) (2016)
* **JAN**: [Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/abs/1605.06636) (2016)

##### NMT
解决finetune时的overfitting问题，mixed finetuning/知识蒸馏/正则化(tuneout):
* [Fine-tuning for neural machine translation with limited degradation
across in- and out-of-domain data]() (2017)
    - 基于知识蒸馏的方式，保持源域模型的参数分布，以避免在源领域的性能下降
* **tuneout**: [Regularization techniques for fine-tuning in neural machine translation]() (EMNLP2017)
    - 为了解决finetuing中的过拟合问题，采用dropout和l2正则化，引入了`tuneout`作为dropout的变量

### Model-based Methods

#### Embedding methods
* [Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/abs/1804.08328v1) (2018)
    - 给每个任务学习相应的embedding，可参考此方式为领域学习embedding.
* **Domain Control (DC)**: [Domain control for neural machine translation]() (RANLP2017)
    - 基于tf-idf给每个word添加domain tag，原表示拼接领域表示去解码。 

#### Adversarial methods
##### General
* **DANN**: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) (2015)
    - 对抗领域适应开山之作，GRL的出处
* **CoGAN**: [Coupled Generative Adversarial Networks](https://arxiv.org/abs/1606.07536) (2016)
* **ADDA**: [Adaptative Discriminative Domain Adaptation](https://arxiv.org/abs/1702.05464) (2017)
* **CyCADA**: [CyCADA: Cycle-Consistent Adversarial Domain Adaptation](http://proceedings.mlr.press/v80/hoffman18a/hoffman18a.pdf) (2017)
##### NMT
* [Effective domain mixing for neural machine translation]() (2017)
    - 共享encoder，引入判别器提取共有特征
* [Multi-domain neural machine translation with wordlevel domain context discrimination]() (EMNLP2018)
    - 共享-私有encoder，引入判别器提取共有特征
* [Improving Domain Adaptation Translation with Domain Invariant and Specific Information]() (ACL2019)
    - 共享-私有encoder & decoder，引入判别器提取共有特征

## Semi-supervised Domain Adaptation for NMT
* [Semi-supervised learning for neural machine translation]() (ACL2016)

## Unsupervised Domain Adaptation for NMT
#### General
* **SA**: [Unsupervised Visual Domain Adaptation Using Subspace Alignment](https://pdfs.semanticscholar.org/51a4/d658c93c5169eef7568d3d1cf53e8e495087.pdf) (2015)
* **DSN**: [Domain Separation Networks](https://arxiv.org/abs/1608.06019) (2016)

#### NMT
* [Investigations on translation model adaptation using monolingual data]() (WMT2011)
    - self-enhancing
* [On using monolingual corpora in neural machine translation]() (2015)
    - 用目标端的单语语料训练RNNLM，在解码时使用
* [Improving neural machine translation models with monolingual data]() (2015)
    - 目标域目标端单语语料做back translation
* [Exploiting source-side monolingual data in neural machine
translation]() (EMNLP2016)
    - self-learning得到目标域伪平行语料
* [Using targetside monolingual data for neural machine translation
through multi-task learning]() (EMNLP2017)
    - RNNLM和翻译模型一起训练


## Multi-source Domain Adaptation for NMT
#### General
* [Domain adaptation from multiple sources via auxiliary classifiers](https://lms.comp.nus.edu.sg/sites/default/files/publication-attachments/icml09-xudong.pdf) (ICML2007)
    - 按照预先定义的源域和目标域的相似性进行加权
* **DSM**: [Domain Selection Machine]() (CVPR2012)
    - 为每个样本寻找与其最近似的源域
* [Domain attention with an ensemble of experts](https://aclweb.org/anthology/P17-1060) (ACL2017)
    - 利用目标域标注数据学习每个领域到目标域样本的attention权重 
* [Deep Cocktail Network: Multi-source Unsupervised Domain Adaptation with Category Shift](www.linliang.net/wp-content/uploads/2018/03/CVPR2018_Cocktail.pdf) (CVPR2018)
    - 关注源域的标签偏移现象，利用多路对抗技术，利用对抗得到的perplexity作为结果综合的权重
* **MoE**: [Multi-Source Domain Adaptation with Mixture of Experts](https://arxiv.org/pdf/1809.02256.pdf) (EMNLP2018)
    - 利用meta-learning无监督学习sample-to-domain metric
* [Moment Matching for Multi-Source Domain Adaptation](https://arxiv.org/abs/1812.01754) (arxiv2018)
    - 源域和目标域分布对齐，最终结果为各源域平均均均
* [Adversarial Multiple Source Domain Adaptation](https://www.cs.cmu.edu/~hzhao1/papers/NIPS2018/nips2018_main.pdf) (NIPS2018)
    - 某源域的error越大，在集成过程中的权重越大
* **MADA**: [Multi-Adversarial Domain Adaptation](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17067/16644) (2018)
* **MDAN**: [Multiple Source Domain Adaptation with Adversarial Learning](https://arxiv.org/abs/1705.09684) (2018)

#### NMT
* [Domain control for neural machine translation](https://arxiv.org/pdf/1612.06140) (RANLP2017)
    - word embedding + domain embedding拼接
* [Multi-domain neural machine translation through unsupervised adaptation]() (WMT2017)
    - 根据test和train的相似性，调节训练算法的超参
* [Neural machine translation training in a multi-domain scenario]() (2017)
    - 研究了data concatenation (源域数据混合), model stacking （在每个领域交替训练模型）, data selection （选择和目标域接近的源域数据） and multi-model ensemble （集成多个单源域模型），他们发现finetuing the concatenation system on in-domain data效果最好
* [An empirical comparison of domain adaptation methods for neural machine translation]() (ACL2017)
    - mixed-finetuning with domain tags, 先用源域数据训练模型得到目标域模型的初始化，再用源域数据和过采样的目标域数据finetuning（数据带领域标记）
* [Effective domain mixing for neural machine translation]() (WMT2017)
    - 提出了三种数据混合的方案：基于领域分类的discriminative mixing，基于对抗的discriminative mixing，以及在目标端添加领域token的target token mixing
* [Multi-domain neural machine translation]() (2018)
    - 将每个领域看作一个语言对，使用multi-lingual的方式实现multi-domain NMT
* [Multilingual and multi-domain adaptation for neural machine translation]() (2018)
* [Multi-domain NMT with word-level context discrimination](https://www.aclweb.org/anthology/D18-1041) (EMNLP2018)
    - 对于多领域混合的语料，在encoder分别学习公有annotation和私有annotation，在deocoder端根据公有和私有context解码

## Mix-domain Domain Adaptation
* [Discovering latent domains for multisource domain adaptation]() (ECCV2012)
* [Latent domain translation models in mix-of-domains haystack]() (COLING2014)
* [Cross-domain Text Classification with Multiple Domains and Disparate
Label Sets]() (ACL2016)
* [Boosting Domain Adaptation by Discovering Latent Domains]() (CVPR2018)

# Datasets
