# NeuroPred

## Environment Setup
1. 创建conda环境
```
conda create -n neuropred python==3.10
```
2.确定当前设备的GPU cuda版本，去pytorch官网 https://pytorch.org/ 查找对应自己环境的pytorch下载命令。以Windows环境，CUDA版本为12.8对应的下载命令为例:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
3.下载调用DNN所需的packages:timm, transformers, diffusers, open_clip, clip
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
4.下载一些辅助性的packages: matplotlib, scikit-learn, accelerate
```
pip install matplotlib
pip install scikit-learn
pip install accelerate
```
## Usage of scripts
一个标准的分析流程：激活 `neuropred` 环境，运行 `GetInfo.ipynb` 获取想要使用的网络及可调用的层；运行 `LayerSearch.py` 在指定模型和层列表中寻找最优编码层；运行 `MEI_pipeline.py` 在 BrainDiVE 框架下生成 MEI；运行 `MEI_review.ipynb` 对不同 Encoder 的 MEI 预测进行评估。运行代码所需的参数通常位于文件开头并附有注释。以下简要介绍各脚本的主要功能及所需文件。

#### Getinfo.ipynb
包含三个单元格：
- 根据 backbone 类型列出所有可用模型
- 根据模型名称返回可调用的层名称
- 根据模型名称和层名称获取特征的形状

此外，Models and Layers.pdf中提供了一些常用模型的结构与可调用层的名称，以供参考。

#### LayerSearch.py
给定神经反应矩阵与对应的图像刺激，构建由指定模型及层列表组成的 Encoder，计算各 Encoder 的编码准确度，并输出最优层。需在脚本开头指定：
- 搜索结果保存目录
- 图像刺激文件夹路径
- 神经反应 `.npz` 文件（包含反应与 noise ceiling）

如需修改读取逻辑，请调整 “1. Load neural responses” 部分。

#### MEI_pipeline.py
给定神经反应矩阵与对应的图像刺激，构建指定模型及层的 Encoder，并与 BrainDiVE 架构连接，生成目标神经元或神经群的 MEI。所需文件与 `LayerSearch.py` 相同，同时需指定 MEI 保存目录。

#### MEI_review.ipynb
对生成的 MEI 图像，使用原始 Encoder 及其他 Encoder 预测对应的神经反应，评估各 Encoder 对 MEI 的预测性能。除以上三个路径外，还需指定存放 MEI 图像的文件夹路径。

