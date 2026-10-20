# 课堂音视频报告：Omni + M2KT 记忆适配器

当前只保留这一版课堂报告模型：Qwen2.5-Omni-3B 基座、课堂中文报告 LoRA，以及借鉴 M2KT 可学习记忆思路的音视频特征适配器。推理默认输出一段中文报告。适配器没有直接调用原医学 M2KT 网络，也没有外部课堂知识库。

## 模型结构

```mermaid
flowchart LR
    Clips["课堂音视频 MP4<br/>每门课两段、每段约 8 秒"] --> Enc["Qwen2.5-Omni 音频与视频编码器<br/>冻结"]
    Enc --> AV["音视频 token<br/>2048 维"]

    subgraph Memory["M2KT 风格记忆适配器 · 唯一训练的模块"]
        direction TB
        Proj["投影到 256 维<br/>加入模态与片段位置表示"] --> Slots["16 个可学习记忆槽<br/>注意力读取音视频证据"]
        Slots --> Read["音视频 token 读取记忆"]
        Read --> Gate["门控残差写回<br/>仅修改音视频 token"]
    end

    AV --> Proj
    AV --> Merge["增强后的音视频 token"]
    Gate --> Merge
    Prompt["原训练提示词<br/>文本 token 不修改"] --> Thinker["Omni Thinker + 课堂报告 LoRA<br/>均冻结"]
    Merge --> Thinker
    Thinker --> Raw["原始中文报告"] --> Format["去栏目标题、合并为一段"] --> Report["课堂报告草稿"]
    Target["训练集原始中文报告"] --> Loss["报告 token 交叉熵"]
    Thinker --> Loss
    Loss -.->|只更新记忆适配器| Memory

    classDef frozen fill:#e8f2ff,stroke:#4d80bd,color:#172536;
    classDef trainable fill:#fff0d4,stroke:#c88324,color:#342514;
    classDef output fill:#e8f6e9,stroke:#47935d,color:#183122;
    class Enc,Thinker frozen;
    class Proj,Slots,Read,Gate trainable;
    class Report output;
```

训练时以原始报告计算损失，只更新记忆适配器；推理时不读取原始报告。最右侧的单段处理仅调整排版，不改变模型权重。

训练数据、运行命令和局限见 [CLASSROOM_M2KT_ADAPTER.md](CLASSROOM_M2KT_ADAPTER.md)。验证集原始报告与生成报告对照见 [validation_examples_single_paragraph.md](results/classroom_m2kt_adapter/validation_examples_single_paragraph.md)。

主要入口：

- `predict_classroom_m2kt_adapter.py`：使用基座、LoRA 和记忆适配器生成报告。
- `classroom_m2kt_adapter.py`：记忆适配器网络。
- `extract_classroom_m2kt_features.py`、`tokenize_classroom_m2kt_targets.py`、`train_classroom_m2kt_adapter.py`：特征缓存、目标缓存与适配器训练。
- `scripts/prepare_classroom_av.py`、`scripts/build_classroom_omni_dataset.py`：准备同步音视频与训练数据。

仓库包含现有代码、依赖清单、课堂标注与原始文字报告、处理后的短音视频片段、训练数据清单、特征与目标缓存，以及实验元数据和验证报告。完整原始课堂录像、Omni 基座权重、课堂 LoRA 权重、记忆适配器权重和本地 Python 虚拟环境不上传。新环境运行推理前需自行准备这三套权重并安装 `requirements-omni.txt`；适配器训练也需要 Omni 基座和课堂 LoRA 权重，仓库中的缓存可免去重新提取音视频特征。部分数据清单记录了原机器的绝对路径，换机器运行时需调整路径。许可范围为研究及内部非商业试验。
