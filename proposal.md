# CS6493 NLP Group Project Proposal

## Topic 3: Building Practical LLM Applications with LlamaIndex

## Application Theme: ScholarLens -- An Intelligent Academic Paper QA & Comparison Assistant

---

## 1. Project Overview

本项目旨在使用 LlamaIndex 框架构建 **ScholarLens**，一个面向学术论文的智能问答与对比分析助手。系统可导入多篇 NLP 领域学术论文（PDF）、技术博客（网页）和课程讲义（文本），支持：

- **单论文问答**：针对某篇论文提问，如 *"What is the main contribution of Self-RAG?"*
- **跨论文对比**：综合多篇论文回答，如 *"Compare the retrieval strategies of RAG and Self-RAG"*
- **多轮对话**：支持追问和上下文保持，如 *"Can you explain their differences in more detail?"*

项目将对比多种 LLM 后端的性能表现，通过系统化的评估指标衡量不同分块策略和检索配置下的回答质量。

---

## 2. Motivation & Problem Statement

研究人员在文献综述阶段需要阅读大量论文，手动检索和对比不同方法的核心思想既耗时又容易遗漏关键信息。现有的学术搜索工具（如 Google Scholar、Semantic Scholar）主要解决"找论文"的问题，但无法回答"论文内容相关"的具体问题。

直接使用 LLM 回答学术问题面临两个核心挑战：
1. **知识截止问题**：LLM 的训练数据有时间限制，无法涵盖最新发表的论文
2. **幻觉生成问题**：LLM 可能编造不存在的方法、错误的实验结果或虚假的引用

RAG（Retrieval-Augmented Generation）通过先检索再生成的方式，将回答锚定在真实的论文内容上，有效缓解以上两个问题。LlamaIndex 作为专为 LLM 应用设计的数据框架，提供了丰富的数据连接器、索引策略和检索模块，是构建此类应用的理想工具。

---

## 3. System Architecture Design

### 3.1 Application Type: Academic Paper QA + Cross-Document Comparison + Conversational Agent

我们将构建一个**三阶段递进系统**：

#### Stage 1: Single-Paper QA（核心功能）
- 使用 LlamaIndex 构建 RAG pipeline，针对单篇论文进行问答
- 支持 PDF 论文的自动解析、分块和索引
- 示例：*"What datasets are used in the Self-RAG paper?"*

#### Stage 2: Cross-Paper Comparison（核心创新点）
- 构建跨论文的统一索引，支持多论文综合检索
- 通过 metadata filtering 区分不同论文来源
- 设计专用的对比 prompt template，引导 LLM 结构化对比
- 示例：*"How do RAG and Self-RAG differ in their retrieval mechanisms?"*

#### Stage 3: Conversational Agent（扩展功能）
- 在 QA 基础上添加短期记忆管理（ChatMemoryBuffer）
- 支持多轮学术对话，保持上下文连贯性
- 示例：用户先问某论文方法，再追问实验细节

### 3.2 Data Connectors

集成以下至少两种数据源连接器：

| 数据源类型 | 实现方式 | 具体内容 |
|-----------|---------|---------|
| PDF 学术论文 | `SimpleDirectoryReader` | 课程参考文献（RAG、Self-RAG、CoT、HaluEval 等 10-15 篇） |
| 技术博客 | `BeautifulSoupWebReader` | LlamaIndex 官方博客、Hugging Face 技术文章 |
| 课程材料 | `SimpleDirectoryReader` | 课程 slides 讲义（.txt/.md 格式） |

**论文数据集（初步规划）**：

| 论文 | 主题 | 来源 |
|------|------|------|
| Lewis et al. (2020) | RAG 原始论文 | NeurIPS 2020 |
| Asai et al. (2024) | Self-RAG | ICLR 2024 |
| Wei et al. (2022) | Chain-of-Thought | NeurIPS 2022 |
| Lin et al. (2022) | TruthfulQA | ACL 2022 |
| Li et al. (2023) | HaluEval | EMNLP 2023 |
| Min et al. (2023) | FActScore | EMNLP 2023 |
| Touvron et al. (2023) | Llama 2 | Meta 2023 |
| Jiang et al. (2023) | Mistral 7B | arXiv 2023 |

> 所有论文均来自项目说明书的参考文献，确保与课程内容高度相关。

### 3.3 Chunking Strategies（分块策略对比）

我们将实验以下分块策略：

| 策略 | 参数设置 | 说明 |
|------|---------|------|
| Fixed-size chunking | 256 tokens, 10% overlap | 项目要求的基准方案 |
| Fixed-size chunking | 512 tokens, 10% overlap | 较大分块对比 |
| Fixed-size chunking | 128 tokens, 10% overlap | 较小分块对比 |
| Sentence-based chunking | 按句子边界分块 | 保持语义完整性 |

### 3.4 LLM Backend Comparison（模型对比）

对比至少 2 种 LLM 后端：

| 模型 | 参数规模 | 部署方式 | 说明 |
|------|---------|---------|------|
| Mistral-7B (GPTQ-4bit) | 7B | Ollama 本地部署 | 项目推荐的主力模型 |
| Qwen2.5-3B / 1.5B | 1.5B-3B | Ollama 本地部署 | 轻量级对比模型 |
| Llama 3.2-3B | 3B | Ollama 本地部署 | 备选对比模型 |

> 考虑到计算资源限制，我们优先使用量化模型（GPTQ-4bit）通过 Ollama 进行本地部署。

---

## 4. Capability Evaluation

### 4.1 Evaluation Dataset

我们将构建一个**学术论文问答评测数据集**，包含 **60-80 个测试问题**，分为三个难度层次：

| 类型 | 数量 | 示例 | 难度 |
|------|------|------|------|
| **Factual QA（事实型）** | 30 | *"What benchmark datasets are used in the RAG paper?"* | Easy |
| **Cross-Paper Comparison（跨论文对比）** | 20 | *"What are the key differences between CoT and Self-Consistency?"* | Medium |
| **Reasoning QA（推理综合型）** | 15 | *"Based on the papers, what are the main strategies to reduce LLM hallucinations?"* | Hard |
| **Conversational QA（多轮对话）** | 10 | 先问方法，再追问实验结果和局限性 | Hard |

每个测试问题包含：
- **Question**：问题文本
- **Ground Truth Answer**：人工撰写的参考答案
- **Source Papers**：答案来源论文标注
- **Question Type**：问题类型标签

### 4.2 Evaluation Metrics

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **Response Relevance** | 回答与问题的相关程度 | LLM-as-Judge（使用 GPT 评分 1-5） |
| **Faithfulness** | 回答是否基于检索到的文档 | 检查回答中的信息是否来源于 context |
| **Task Completion Rate** | 成功回答问题的比例 | 正确回答数 / 总问题数 |
| **Response Latency** | 系统响应时间 | 端到端延迟（秒） |
| **Retrieval Precision** | 检索到的文档片段的相关性 | 相关片段数 / 总检索片段数 |

### 4.3 Ablation Study（消融实验）

我们将从以下维度进行消融分析：

1. **分块策略对比**：不同 chunk size 对检索质量和回答准确性的影响
2. **模型对比**：不同 LLM 后端在相同 pipeline 下的表现差异
3. **Top-K 检索数量**：检索 1/3/5/10 个文档片段时的性能变化
4. **Embedding 模型对比**：不同 embedding 模型对检索质量的影响

---

## 5. Technical Implementation Plan

### 5.1 Technology Stack

```
Core Framework:    LlamaIndex (Python)
LLM Deployment:    Ollama (local inference)
Embedding Model:   BAAI/bge-small-en-v1.5 or sentence-transformers
Vector Store:      Chroma / FAISS (local)
Frontend (可选):   Gradio / Streamlit
Language:          Python 3.10+
```

### 5.2 Project Structure

```
ScholarLens/
├── data/
│   ├── papers/              # 学术论文 PDF（RAG, Self-RAG, CoT 等）
│   ├── blogs/               # 技术博客 URL 列表
│   ├── course_materials/    # 课程讲义
│   └── eval/                # 评测数据集
│       ├── questions.json   # 测试问题 + ground truth
│       └── README.md        # 数据集说明
├── src/
│   ├── indexing/
│   │   ├── data_loader.py   # 多源数据连接器
│   │   ├── chunking.py      # 分块策略（4种）
│   │   └── metadata.py      # 论文元数据提取（标题、作者、年份）
│   ├── retrieval/
│   │   └── retriever.py     # 检索器（含 metadata filtering）
│   ├── generation/
│   │   ├── llm_config.py    # LLM 后端配置
│   │   └── prompts.py       # Prompt Templates（含对比 prompt）
│   ├── agent/
│   │   └── chat_agent.py    # 对话式 Agent + Memory
│   └── evaluation/
│       ├── metrics.py        # 评估指标实现
│       └── evaluator.py      # 批量自动评估器
├── notebooks/
│   ├── 01_data_preparation.ipynb   # 数据导入与预处理
│   ├── 02_single_paper_qa.ipynb    # 单论文问答 Demo
│   ├── 03_cross_paper_compare.ipynb # 跨论文对比 Demo
│   ├── 04_chat_agent.ipynb          # 对话 Agent Demo
│   └── 05_evaluation.ipynb          # 完整评估实验
├── requirements.txt
└── README.md
```

---

## 6. Timeline & Milestones

| 阶段 | 时间 | 任务 | 交付物 |
|------|------|------|--------|
| **Phase 1** | Week 1 (Mar 9-15) | 环境搭建 + 数据收集 | Ollama 部署完成，测试文档收集 |
| **Phase 2** | Week 2 (Mar 16-22) | 核心 RAG Pipeline 开发 | Document QA 基本功能可用 |
| **Phase 3** | Week 3 (Mar 23-25) | 评测数据集构建 + 进度报告 | **Progress Report 提交 (3/25)** |
| **Phase 4** | Week 4 (Mar 26-Apr 1) | 分块策略实验 + 模型对比 | 实验结果初步分析 |
| **Phase 5** | Week 5 (Apr 2-8) | Conversational Agent + 消融实验 | Agent 功能完成 + 完整实验数据 |
| **Phase 6** | Week 6 (Apr 9-14) | 报告撰写 + 演示准备 | **Presentation (4/14)** |
| **Final** | Apr 15 | 最终提交 | **Final Report + Code (4/15)** |

---

## 7. Division of Labor（分工建议）

> 以下为建议分工模板，可根据实际组员人数（1-6人）调整：

| 角色 | 主要任务 |
|------|---------|
| **成员 A** | 项目架构设计 + RAG Pipeline 核心开发 |
| **成员 B** | 数据收集 + 数据连接器实现 + 分块策略实验 |
| **成员 C** | LLM 部署 (Ollama) + 模型对比实验 |
| **成员 D** | 评测数据集构建 + 评估指标实现 |
| **成员 E** | Conversational Agent 开发 |
| **成员 F** | 报告撰写 + 演示 PPT 制作 |

---

## 8. Expected Contributions & Innovation

1. **跨论文对比问答**：区别于单文档 QA 的常规实现，我们引入跨论文对比功能，通过 metadata filtering + 对比 prompt template 实现结构化的论文方法对比，这是本项目的**核心创新点**
2. **面向学术场景的 RAG 优化**：针对学术论文的特点（公式多、表格多、引用关系复杂），探索适合论文场景的分块策略
3. **多维度分块策略对比**：系统化比较 4 种分块策略在学术问答场景下的表现差异
4. **轻量级部署方案**：在有限计算资源下，通过量化模型实现实用的学术助手
5. **可复现的评估框架**：构建包含多维度指标的学术 QA 评测数据集和自动化评估 pipeline

---

## 9. Risk Assessment & Mitigation

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| 计算资源不足 | 无法运行大模型 | 使用 GPTQ-4bit 量化 + 小模型 (1.5B-3B) |
| LlamaIndex API 变更 | 代码兼容性问题 | 固定版本号，参考官方文档 |
| 评测数据集质量 | 实验结果不可靠 | 人工审核 + 交叉验证 |
| 时间紧张 | 功能不完整 | 优先完成核心 QA 功能，Agent 作为扩展 |

---

## 10. Demo Scenarios（演示场景设计）

在 4 月 14 日的 15 分钟演示中，我们计划展示以下场景：

**场景 1 - 单论文问答（2 分钟）**
> 用户: "What is the main idea of Self-RAG?"
> ScholarLens: "Self-RAG proposes a framework that learns to retrieve, generate, and critique through self-reflection. Unlike standard RAG which retrieves for every query, Self-RAG trains the LM to adaptively retrieve passages on-demand..." [Source: Asai et al., ICLR 2024]

**场景 2 - 跨论文对比（3 分钟）**
> 用户: "Compare the retrieval strategies used in RAG and Self-RAG"
> ScholarLens: "RAG (Lewis et al., 2020) uses a fixed retrieve-then-generate pipeline where retrieval is performed for every input. In contrast, Self-RAG (Asai et al., 2024) introduces reflection tokens that allow the model to decide when to retrieve..."

**场景 3 - 多轮对话（2 分钟）**
> 用户: "Tell me about Chain-of-Thought prompting"
> ScholarLens: [回答 CoT 概述]
> 用户: "How does it compare with Self-Consistency?"
> ScholarLens: [基于上下文进行对比]

**场景 4 - 实验结果展示（8 分钟）**
展示不同分块策略、不同模型、不同 Top-K 下的性能对比图表。

---

## 11. References

1. Touvron H, et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models." Meta 2023.
2. Jiang W, et al. "Mistral 7B." arXiv preprint arXiv:2310.06825 (2023).
3. Xiao G, et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML 2023.
4. LlamaIndex OSS Documentation: https://developers.llamaindex.ai/python/framework
5. Lewis P, et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks." NeurIPS 2020.
6. Gao Y, et al. "Retrieval-Augmented Generation for Large Language Models: A Survey." arXiv preprint arXiv:2312.10997, 2023.
7. Asai A, et al. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR 2024.
8. Wei J, et al. "Chain-of-thought prompting elicits reasoning in large language models." NeurIPS 2022.
