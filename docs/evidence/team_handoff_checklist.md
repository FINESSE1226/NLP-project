# ScholarLens 小组进度与分工清单（组员传阅）

> 仓库：[FINESSE1226/NLP-project](https://github.com/FINESSE1226/NLP-project)  
> 默认开发分支建议：`caoxueqi_dev`（或个人 `feature/*`）→ PR 合并到 `main`  
> 环境说明见根目录 [`README.md`](../README.md)

---

## 一、当前已完成内容（可直接使用）

### 1. 文档与约定

| 项 | 说明 |
|----|------|
| `proposal.md` | 项目开题与规划（中文全文已恢复 UTF-8，可正常阅读） |
| `docs/contract.md` | 元数据 / 评测数据 JSON 等约定 |
| `docs/evidence/sample_run_log.txt` | 本地跑通记录的示例输出（留档用） |
| `README.md` | 环境、`Phase 1/2` 命令、目录说明、组员 Handoff 步骤 |
| `configs/default.example.yaml` | 参数参考（CLI 脚本当前以命令行参数为主，YAML 为参考） |

### 2. 代码与脚本（Phase 1–2 底座）

| 项 | 说明 |
|----|------|
| `scholarlens/manifest.py` | 读取并校验 `data/papers/manifest.csv` |
| `scholarlens/ollama_config.py` | Ollama LLM / 嵌入 / 分块（SentenceSplitter） |
| `scholarlens/indexing.py` | 按 manifest 载入 PDF、写入 `paper_id` 等元数据，建索引并持久化 |
| `scripts/check_env.py` | 环境自检（无 LLM 调用） |
| `scripts/minimal_rag_smoke.py` | Phase 1：单目录 PDF → 内存索引 → 单轮问答 |
| `scripts/build_index.py` | Phase 2：持久化建库 → `storage/index/`（默认，已 `.gitignore`） |
| `scripts/query_index.py` | Phase 2：加载索引；支持 `--paper-id`、`--top-k` 等 |
| `requirements.txt` | Python 依赖 |
| `.gitignore` | 忽略 `.venv`、`storage/`、密钥与缓存等 |

### 3. 数据样例

| 项 | 说明 |
|----|------|
| `data/papers/manifest.csv` | 示例论文清单（含 `lewis2020_rag` 等字段） |
| `data/papers/*.pdf`（若仓库内已含） | 示例 PDF；也可按 `data/papers/README.md` 自行下载 |
| GitHub | `main` 已合并 PR #1（ScholarLens 批量提交）；`caoxueqi_dev` 含后续修复（如 `proposal.md` 编码） |

### 4. 与 `proposal.md` 中「计划结构」的差异（重要）

- 当前仓库采用 **`scholarlens/` + `scripts/`**，尚未按 `proposal` 里的 `src/indexing/...` 大目录重构；**功能上已覆盖 Stage 1 单论文 RAG 的核心路径**（建库 + 查询 + 元数据）。
- **第二种数据源**（博客 / 讲义）、**跨论文对比专用流程**、**评测集与批量实验**、**对话 Agent** 等见下文「待完成」。

---

## 二、待完成内容（按优先级）

| 优先级 | 内容 | 交付物建议 |
|--------|------|------------|
| P0 | 扩充 `manifest.csv` + 对应 PDF（目标多篇，如 proposal 10–15 篇量级可分期） | 更新 manifest + `README` 说明 |
| P0 | **第二数据源**（proposal：博客 URL 或 `course_materials` `.md`/`.txt`），至少一种 LlamaIndex Reader 接入 | 数据目录 + 构建脚本或合并进现有 `build_index` 流程 |
| P1 | **跨论文对比**：不限定 `paper_id` 的检索 + 专用对比 Prompt / 可选 CLI 子命令 | `scripts/` 或薄封装 + 示例问题 |
| P1 | **实验**：多种 `chunk_size` / `top_k`、≥ 2 个 LLM（Ollama 多模型） | 表格或 `experiments/` 下脚本 + 结果汇总 |
| P1 | **评测**：`eval/questions.json`（目标 60–80 题，可先 20 题跑通）+ 指标（含 `contract` 约定） | `data/eval/` 或 `eval/` + 批跑脚本 |
| P2 | **对话 Agent** + 短期记忆（ChatMemoryBuffer） | 新脚本或模块 + README 一节 |
| P2 | 可选 **Gradio / Streamlit** 演示界面 | 单独文件 + 依赖说明 |
| 全程 | **课程交付**：进度报告、期末报告、PPT、演示排练 | 按课表 deadline（见 `proposal` 时间表） |

---

## 三、分工建议（6 人组，可改姓名贴在右侧）

> 已与 **proposal §7** 对齐；请组长在「负责人」列填名字，避免多人撞车。

| 角色 | 主要任务 | 与当前仓库的关系 | 负责人（待填） |
|------|----------|------------------|----------------|
| **成员 A** | 项目架构 + **RAG 管线核心** | ** largely DONE**：`scholarlens` + `build_index` / `query_index`；后续可 refactor 对齐 `proposal` 目录或加配置读 YAML | |
| **成员 B** | **数据** + 连接器 + **分块实验** | manifest/PDF 扩充；博客或讲义接入；对比不同 `chunk_size` 并记结果 | |
| **成员 C** | **Ollama / LLM** + **模型对比实验** | 维护多模型 pull 说明；用同一批问题跑 ≥2 模型，填对比表 | |
| **成员 D** | **评测集** + **评估指标 / 自动批跑** | 落地 `eval/questions.json`、`contract`；可选 RAGAS / LLM-as-judge | |
| **成员 E** | **对话 Agent** + 记忆 | 在现索引之上加 Chat Engine；与 `query_index` 并列文档 | |
| **成员 F** | **报告 + PPT + 演示流程** | 与 A–E 同步截图、数字、demo 脚本；不拖 deadline | |

**协作约定（建议）**

- 个人开发用 **`feature/姓名-任务`** 或继续使用 **`caoxueqi_dev`** 约定短周期，合并 **`main`** 走 **Pull Request**。
- 克隆后务必：`pip install -r requirements.txt` → 自备 PDF → `python scripts/check_env.py` → `build_index` → `query_index`。

---

## 四、时间轴提醒（摘自 `proposal.md`，以课程最新通知为准）

| 节点 | 内容 |
|------|------|
| Week 3 | 进度报告（示例 **3/25**） |
| Week 6 | 课堂演示（示例 **4/14**） |
| Final | 期末报告 + 代码（示例 **4/15**） |

---

## 五、一键自检（组员接到任务后）

```bash
python scripts/check_env.py
python scripts/build_index.py
python scripts/query_index.py "What is retrieval-augmented generation?"
```

若失败：先看 `README.md` 与 Ollama 是否已拉取嵌入模型与对话模型。

---

*本清单由项目组维护；更新时请同步提交到仓库 `docs/` 或备注在 PR 说明中。*
