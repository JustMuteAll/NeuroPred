# NeuroPred

## Environment Setup

## Usage of scripts
一个标准的分析流程：激活 `NeuroPredictor` 环境，运行 `GetInfo.ipynb` 获取想要使用的网络及可调用的层；运行 `LayerSearch.py` 在指定模型和层列表中寻找最优编码层；运行 `MEI_pipeline.py` 在 BrainDiVE 框架下生成 MEI；运行 `MEI_review.ipynb` 对不同 Encoder 的 MEI 预测进行评估。运行代码所需的参数通常位于文件开头并附有注释。以下简要介绍各脚本的主要功能及所需文件。

#### Getinfo.ipynb
包含三个单元格：
- 根据 backbone 类型列出所有可用模型
- 根据模型名称返回可调用的层名称
- 根据模型名称和层名称获取特征的形状

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

