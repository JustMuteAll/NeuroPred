# NeuroPred

## Environment Setup

1. 创建conda环境
```
conda create -n neuropred python==3.10
conda activate neuropred
```
 
 2. 确定当前设备的GPU cuda版本，去pytorch官网 https://pytorch.org/ 查找对应自己环境的pytorch下载命令。以Windows环境，CUDA版本为12.8对应的下载命令为例:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3. 下载调用DNN所需的packages:timm, transformers, diffusers, open_clip, clip
```
pip install timm
pip install transformers
pip install diffusers
pip install open_clip_torch
```
clip的下载直接使用"pip install clip" 有时会出现一些bug，通常我会使用: 
```
pip install git+https://github.com/openai/CLIP.git
```

4. 下载一些辅助性的packages: matplotlib, scikit-learn, accelerate
```
pip install matplotlib
pip install scikit-learn
pip install accelerate
```
## Usage of scripts

运行代码所需的参数通常位于文件开头并附有注释。以下简要介绍各脚本的主要功能及所需文件。
#### DataProcessing.py

给定一个处理后的 GoodUnit 文件，或一个包含多个 GoodUnit 的文件夹，对数据进行处理并生成含有反应矩阵与 noise ceiling 的 `.npz` 文件，可作为后续 Encoding 计算的输入。
#### Getinfo.ipynb

包含三个单元格：
- 根据 backbone 类型列出所有可用模型
- 根据模型名称返回可调用的层名称
- 根据模型名称和层名称获取特征的形状

此外，Models and Layers.pdf中提供了一些常用模型的结构与可调用层的名称，以供参考。
#### FeatureExtracting.py

在给定 backbone 类型、模型名称、特征层与待处理图像文件夹后，提取对应的特征。若不作额外设置，默认使用预训练权重；在确认权重与网络结构匹配后，可通过 `ckpt_path` 引入外部权重。

#### LayerSearch.py

给定神经反应矩阵与对应的图像刺激，构建由指定模型及层列表组成的 Encoder，计算各 Encoder 的编码准确度，并输出最优层。需在脚本开头指定：
- 搜索结果保存目录
- 图像刺激文件夹路径
- 神经反应 `.npz` 文件（包含神经反应与 noise ceiling）

#### MEI_pipeline.py

给定神经反应矩阵与对应的图像刺激，构建指定模型及层的 Encoder，并与 BrainDiVE 架构连接，生成目标神经元或神经群的 MEI。所需文件与 `LayerSearch.py` 相同，同时需指定 MEI 保存目录。

#### MEI_review.ipynb

对生成的 MEI 图像，使用原始 Encoder 及其他 Encoder 预测对应的神经反应，评估各 Encoder 对 MEI 的预测性能。除以上三个路径外，还需指定存放 MEI 图像的文件夹路径。

#### Visualization.py

指定 backbone 类型、模型名、目标特征层与图像目录后，批量提取该层特征并对单一神经元进行编码建模，随后将提取器与线性读出拼接为可视化模型，基于 SmoothGrad/GradCAM 生成并保存每张图像的热力图，同时统计并绘制平均预测响应曲线。默认使用预处理权重（`ckpt_path=None`），预处理与设备由提取器自动配置。


