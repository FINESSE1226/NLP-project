# 个人贡献说明（成员 A / 技术主干）

> 项目：ScholarLens — CS6493 Topic 3（LlamaIndex 学术问答）  
> 仓库：[FINESSE1226/NLP-project](https://github.com/FINESSE1226/NLP-project)  
> 说明：以下内容归纳**本人**在本仓库中已交付的工作；与 proposal 等文稿中「小组共创」重叠处，以**工程实现与 Git 提交**为准。

---

## 一、核心代码与包结构

| 路径 | 内容 |
|------|------|
| `scholarlens/__init__.py` | 包版本等元信息 |
| `scholarlens/manifest.py` | 读取并校验 `data/papers/manifest.csv`，解析 `PaperRecord`，解析 PDF 路径 |
| `scholarlens/ollama_config.py` | 统一配置 LlamaIndex `Settings`：Ollama LLM、Ollama 嵌入、`SentenceSplitter`（chunk 参数） |
| `scholarlens/indexing.py` | 按 manifest 载入 PDF、写入 `paper_id` / `title` / `year` / `source_url` / `file_name` 等元数据，构建 `VectorStoreIndex` 并持久化 |

---

## 二、命令行脚本（Phase 1 / Phase 2）

| 路径 | 内容 |
|------|------|
| `scripts/check_env.py` | 环境自检：Python、`llama_index`、Ollama 连通与已安装模型、`data/papers` 下 PDF、manifest 是否存在（无 LLM 调用） |
| `scripts/minimal_rag_smoke.py` | Phase 1：单目录 PDF → 内存索引 → 单轮问答（适配 LlamaIndex 0.14+，`llama_index.core` 等） |
| `scripts/build_index.py` | Phase 2：从 manifest 建持久化索引至 `storage/index/`（默认路径，目录 gitignore） |
| `scripts/query_index.py` | Phase 2：加载持久化索引并问答；支持 `--paper-id` 元数据过滤、`--top-k`、模型与 `base-url` 等参数 |

---

## 三、项目配置与依赖

| 路径 | 内容 |
|------|------|
| `requirements.txt` | `llama-index`、Ollama 集成、`python-dotenv` 等依赖声明 |
| `configs/default.example.yaml` | Ollama 与分块等**参考**配置（脚本以 CLI 为主时可对照填写） |
| `.gitignore` | 排除 `.venv`、`storage/`、密钥、缓存、`*.log` 等，避免误提交环境与索引 |

---

## 四、文档与数据约定

| 路径 | 内容 |
|------|------|
| `README.md` | 环境准备、Phase 1/2 说明、目录说明、从仓库根运行脚本、YAML 与 CLI 关系、**组员 Handoff 自检**、分工清单链接 |
| `docs/contract.md` | 元数据 / 评测 JSON 等约定 |
| `docs/evidence/sample_run_log.txt` | 本地跑通 `check_env` → `build_index` → `query_index` 的示例输出留档 |
| `docs/team_handoff_checklist.md` | 小组进度、待办、分工（成员 A–F）与自检命令 |
| `data/papers/manifest.csv` | 示例论文清单字段 |
| `data/papers/README.md` | 数据目录说明 |
| `data/papers/.gitkeep` | 保留空目录结构（按需） |

---

## 五、Git / GitHub 与协作（本人执行）

- 在 **`caoxueqi_dev`** 上维护 ScholarLens 相关提交与推送。
- 处理与 **`origin/main`** 合并时 **`README.md` 的冲突**：保留 ScholarLens 完整说明并修正编码显示问题。
- **`proposal.md`**：自 Git 历史恢复 **UTF-8 中文正文**，修正乱码并提交。
- 通过 **Pull Request** 将功能合入 **`main`**（如 PR #1）。
- 新增 **`docs/team_handoff_checklist.md`**，并在 `README.md` 中加入口链接。

**说明**：`proposal.md` 正文通常为**小组开题共同成果**；本人在该文件上的**可独立核实贡献**为：**恢复可读 UTF-8 版本并提交**。

---

## 六、与课程 proposal 的对应关系

- **已交付**：Stage 1 方向的 **单论文 RAG 管线**（PDF + manifest 元数据 + 持久化索引 + CLI 查询 + `paper_id` 过滤），对应 proposal 中「核心 RAG Pipeline」的**可运行底座**。
- **不在本人单独交付范围内**（组内其他分工或后续迭代）：第二数据源、跨论文专用对比流程、系统化实验与评测集、对话 Agent、可选前端等，见 [`team_handoff_checklist.md`](team_handoff_checklist.md)。

---

## 七、向助教 / 组员佐证时可引用

- Git **提交记录**中本人账号下对 `scholarlens/`、`scripts/`、`README.md`、`docs/` 等的改动。
- [`README.md`](../README.md) 与 [`docs/evidence/sample_run_log.txt`](evidence/sample_run_log.txt) 中的复现步骤与示例输出。
- [`docs/team_handoff_checklist.md`](team_handoff_checklist.md) 中对「成员 A」与已完成条目的描述。

---

*可附于进度报告或期末报告「个人贡献」小节。*
