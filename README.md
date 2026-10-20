# 课堂音视频报告：Omni + M2KT 记忆适配器

当前只保留这一版课堂报告模型：Qwen2.5-Omni-3B 基座、课堂中文报告 LoRA，以及借鉴 M2KT 可学习记忆思路的音视频特征适配器。推理默认输出一段中文报告。适配器没有直接调用原医学 M2KT 网络，也没有外部课堂知识库。

## 模型结构

![课堂音视频报告模型结构图](docs/classroom_m2kt_architecture.png)

图中分别给出了推理路径、记忆适配器内部结构和训练监督。可运行 `python scripts/draw_classroom_m2kt_architecture.py` 重新生成 PNG；绘图需要 Matplotlib 和 Noto Sans CJK 字体。

训练时以原始报告计算损失，只更新记忆适配器；推理时不读取原始报告。最右侧的单段处理仅调整排版，不改变模型权重。

训练数据、运行命令和局限见 [CLASSROOM_M2KT_ADAPTER.md](CLASSROOM_M2KT_ADAPTER.md)。验证集原始报告与生成报告对照见 [validation_examples_single_paragraph.md](results/classroom_m2kt_adapter/validation_examples_single_paragraph.md)。

主要入口：

- `predict_classroom_m2kt_adapter.py`：使用基座、LoRA 和记忆适配器生成报告。
- `classroom_m2kt_adapter.py`：记忆适配器网络。
- `extract_classroom_m2kt_features.py`、`tokenize_classroom_m2kt_targets.py`、`train_classroom_m2kt_adapter.py`：特征缓存、目标缓存与适配器训练。
- `scripts/prepare_classroom_av.py`、`scripts/build_classroom_omni_dataset.py`：准备同步音视频与训练数据。

仓库包含现有代码、依赖清单、课堂标注与原始文字报告、处理后的短音视频片段、训练数据清单、特征与目标缓存，以及实验元数据和验证报告。完整原始课堂录像、Omni 基座权重、课堂 LoRA 权重、记忆适配器权重和本地 Python 虚拟环境不上传。新环境运行推理前需自行准备这三套权重并安装 `requirements-omni.txt`；适配器训练也需要 Omni 基座和课堂 LoRA 权重，仓库中的缓存可免去重新提取音视频特征。部分数据清单记录了原机器的绝对路径，换机器运行时需调整路径。许可范围为研究及内部非商业试验。
