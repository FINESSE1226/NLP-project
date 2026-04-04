# CS6493 NLP Group Project Proposal

## Topic 3: Building Practical LLM Applications with LlamaIndex

## Application Theme: ScholarLens -- An Intelligent Academic Paper QA & Comparison Assistant

---

## 1. Project Overview

鏈」鐩棬鍦ㄤ娇鐢?LlamaIndex 妗嗘灦鏋勫缓 **ScholarLens**锛屼竴涓潰鍚戝鏈鏂囩殑鏅鸿兘闂瓟涓庡姣斿垎鏋愬姪鎵嬨€傜郴缁熷彲瀵煎叆澶氱瘒 NLP 棰嗗煙瀛︽湳璁烘枃锛圥DF锛夈€佹妧鏈崥瀹紙缃戦〉锛夊拰璇剧▼璁蹭箟锛堟枃鏈級锛屾敮鎸侊細

- **鍗曡鏂囬棶绛?*锛氶拡瀵规煇绡囪鏂囨彁闂紝濡?*"What is the main contribution of Self-RAG?"*
- **璺ㄨ鏂囧姣?*锛氱患鍚堝绡囪鏂囧洖绛旓紝濡?*"Compare the retrieval strategies of RAG and Self-RAG"*
- **澶氳疆瀵硅瘽**锛氭敮鎸佽拷闂拰涓婁笅鏂囦繚鎸侊紝濡?*"Can you explain their differences in more detail?"*

椤圭洰灏嗗姣斿绉?LLM 鍚庣鐨勬€ц兘琛ㄧ幇锛岄€氳繃绯荤粺鍖栫殑璇勪及鎸囨爣琛￠噺涓嶅悓鍒嗗潡绛栫暐鍜屾绱㈤厤缃笅鐨勫洖绛旇川閲忋€?
---

## 2. Motivation & Problem Statement

鐮旂┒浜哄憳鍦ㄦ枃鐚患杩伴樁娈甸渶瑕侀槄璇诲ぇ閲忚鏂囷紝鎵嬪姩妫€绱㈠拰瀵规瘮涓嶅悓鏂规硶鐨勬牳蹇冩€濇兂鏃㈣€楁椂鍙堝鏄撻仐婕忓叧閿俊鎭€傜幇鏈夌殑瀛︽湳鎼滅储宸ュ叿锛堝 Google Scholar銆丼emantic Scholar锛変富瑕佽В鍐?鎵捐鏂?鐨勯棶棰橈紝浣嗘棤娉曞洖绛?璁烘枃鍐呭鐩稿叧"鐨勫叿浣撻棶棰樸€?
鐩存帴浣跨敤 LLM 鍥炵瓟瀛︽湳闂闈复涓や釜鏍稿績鎸戞垬锛?1. **鐭ヨ瘑鎴闂**锛歀LM 鐨勮缁冩暟鎹湁鏃堕棿闄愬埗锛屾棤娉曟兜鐩栨渶鏂板彂琛ㄧ殑璁烘枃
2. **骞昏鐢熸垚闂**锛歀LM 鍙兘缂栭€犱笉瀛樺湪鐨勬柟娉曘€侀敊璇殑瀹為獙缁撴灉鎴栬櫄鍋囩殑寮曠敤

RAG锛圧etrieval-Augmented Generation锛夐€氳繃鍏堟绱㈠啀鐢熸垚鐨勬柟寮忥紝灏嗗洖绛旈敋瀹氬湪鐪熷疄鐨勮鏂囧唴瀹逛笂锛屾湁鏁堢紦瑙ｄ互涓婁袱涓棶棰樸€侺lamaIndex 浣滀负涓撲负 LLM 搴旂敤璁捐鐨勬暟鎹鏋讹紝鎻愪緵浜嗕赴瀵岀殑鏁版嵁杩炴帴鍣ㄣ€佺储寮曠瓥鐣ュ拰妫€绱㈡ā鍧楋紝鏄瀯寤烘绫诲簲鐢ㄧ殑鐞嗘兂宸ュ叿銆?
---

## 3. System Architecture Design

### 3.1 Application Type: Academic Paper QA + Cross-Document Comparison + Conversational Agent

鎴戜滑灏嗘瀯寤轰竴涓?*涓夐樁娈甸€掕繘绯荤粺**锛?
#### Stage 1: Single-Paper QA锛堟牳蹇冨姛鑳斤級
- 浣跨敤 LlamaIndex 鏋勫缓 RAG pipeline锛岄拡瀵瑰崟绡囪鏂囪繘琛岄棶绛?- 鏀寔 PDF 璁烘枃鐨勮嚜鍔ㄨВ鏋愩€佸垎鍧楀拰绱㈠紩
- 绀轰緥锛?"What datasets are used in the Self-RAG paper?"*

#### Stage 2: Cross-Paper Comparison锛堟牳蹇冨垱鏂扮偣锛?- 鏋勫缓璺ㄨ鏂囩殑缁熶竴绱㈠紩锛屾敮鎸佸璁烘枃缁煎悎妫€绱?- 閫氳繃 metadata filtering 鍖哄垎涓嶅悓璁烘枃鏉ユ簮
- 璁捐涓撶敤鐨勫姣?prompt template锛屽紩瀵?LLM 缁撴瀯鍖栧姣?- 绀轰緥锛?"How do RAG and Self-RAG differ in their retrieval mechanisms?"*

#### Stage 3: Conversational Agent锛堟墿灞曞姛鑳斤級
- 鍦?QA 鍩虹涓婃坊鍔犵煭鏈熻蹇嗙鐞嗭紙ChatMemoryBuffer锛?- 鏀寔澶氳疆瀛︽湳瀵硅瘽锛屼繚鎸佷笂涓嬫枃杩炶疮鎬?- 绀轰緥锛氱敤鎴峰厛闂煇璁烘枃鏂规硶锛屽啀杩介棶瀹為獙缁嗚妭

### 3.2 Data Connectors

闆嗘垚浠ヤ笅鑷冲皯涓ょ鏁版嵁婧愯繛鎺ュ櫒锛?
| 鏁版嵁婧愮被鍨?| 瀹炵幇鏂瑰紡 | 鍏蜂綋鍐呭 |
|-----------|---------|---------|
| PDF 瀛︽湳璁烘枃 | `SimpleDirectoryReader` | 璇剧▼鍙傝€冩枃鐚紙RAG銆丼elf-RAG銆丆oT銆丠aluEval 绛?10-15 绡囷級 |
| 鎶€鏈崥瀹?| `BeautifulSoupWebReader` | LlamaIndex 瀹樻柟鍗氬銆丠ugging Face 鎶€鏈枃绔?|
| 璇剧▼鏉愭枡 | `SimpleDirectoryReader` | 璇剧▼ slides 璁蹭箟锛?txt/.md 鏍煎紡锛?|

**璁烘枃鏁版嵁闆嗭紙鍒濇瑙勫垝锛?*锛?
| 璁烘枃 | 涓婚 | 鏉ユ簮 |
|------|------|------|
| Lewis et al. (2020) | RAG 鍘熷璁烘枃 | NeurIPS 2020 |
| Asai et al. (2024) | Self-RAG | ICLR 2024 |
| Wei et al. (2022) | Chain-of-Thought | NeurIPS 2022 |
| Lin et al. (2022) | TruthfulQA | ACL 2022 |
| Li et al. (2023) | HaluEval | EMNLP 2023 |
| Min et al. (2023) | FActScore | EMNLP 2023 |
| Touvron et al. (2023) | Llama 2 | Meta 2023 |
| Jiang et al. (2023) | Mistral 7B | arXiv 2023 |

> 鎵€鏈夎鏂囧潎鏉ヨ嚜椤圭洰璇存槑涔︾殑鍙傝€冩枃鐚紝纭繚涓庤绋嬪唴瀹归珮搴︾浉鍏炽€?
### 3.3 Chunking Strategies锛堝垎鍧楃瓥鐣ュ姣旓級

鎴戜滑灏嗗疄楠屼互涓嬪垎鍧楃瓥鐣ワ細

| 绛栫暐 | 鍙傛暟璁剧疆 | 璇存槑 |
|------|---------|------|
| Fixed-size chunking | 256 tokens, 10% overlap | 椤圭洰瑕佹眰鐨勫熀鍑嗘柟妗?|
| Fixed-size chunking | 512 tokens, 10% overlap | 杈冨ぇ鍒嗗潡瀵规瘮 |
| Fixed-size chunking | 128 tokens, 10% overlap | 杈冨皬鍒嗗潡瀵规瘮 |
| Sentence-based chunking | 鎸夊彞瀛愯竟鐣屽垎鍧?| 淇濇寔璇箟瀹屾暣鎬?|

### 3.4 LLM Backend Comparison锛堟ā鍨嬪姣旓級

瀵规瘮鑷冲皯 2 绉?LLM 鍚庣锛?
| 妯″瀷 | 鍙傛暟瑙勬ā | 閮ㄧ讲鏂瑰紡 | 璇存槑 |
|------|---------|---------|------|
| Mistral-7B (GPTQ-4bit) | 7B | Ollama 鏈湴閮ㄧ讲 | 椤圭洰鎺ㄨ崘鐨勪富鍔涙ā鍨?|
| Qwen2.5-3B / 1.5B | 1.5B-3B | Ollama 鏈湴閮ㄧ讲 | 杞婚噺绾у姣旀ā鍨?|
| Llama 3.2-3B | 3B | Ollama 鏈湴閮ㄧ讲 | 澶囬€夊姣旀ā鍨?|

> 鑰冭檻鍒拌绠楄祫婧愰檺鍒讹紝鎴戜滑浼樺厛浣跨敤閲忓寲妯″瀷锛圙PTQ-4bit锛夐€氳繃 Ollama 杩涜鏈湴閮ㄧ讲銆?
---

## 4. Capability Evaluation

### 4.1 Evaluation Dataset

鎴戜滑灏嗘瀯寤轰竴涓?*瀛︽湳璁烘枃闂瓟璇勬祴鏁版嵁闆?*锛屽寘鍚?**60-80 涓祴璇曢棶棰?*锛屽垎涓轰笁涓毦搴﹀眰娆★細

| 绫诲瀷 | 鏁伴噺 | 绀轰緥 | 闅惧害 |
|------|------|------|------|
| **Factual QA锛堜簨瀹炲瀷锛?* | 30 | *"What benchmark datasets are used in the RAG paper?"* | Easy |
| **Cross-Paper Comparison锛堣法璁烘枃瀵规瘮锛?* | 20 | *"What are the key differences between CoT and Self-Consistency?"* | Medium |
| **Reasoning QA锛堟帹鐞嗙患鍚堝瀷锛?* | 15 | *"Based on the papers, what are the main strategies to reduce LLM hallucinations?"* | Hard |
| **Conversational QA锛堝杞璇濓級** | 10 | 鍏堥棶鏂规硶锛屽啀杩介棶瀹為獙缁撴灉鍜屽眬闄愭€?| Hard |

姣忎釜娴嬭瘯闂鍖呭惈锛?- **Question**锛氶棶棰樻枃鏈?- **Ground Truth Answer**锛氫汉宸ユ挵鍐欑殑鍙傝€冪瓟妗?- **Source Papers**锛氱瓟妗堟潵婧愯鏂囨爣娉?- **Question Type**锛氶棶棰樼被鍨嬫爣绛?
### 4.2 Evaluation Metrics

| 鎸囨爣 | 璇存槑 | 璁＄畻鏂瑰紡 |
|------|------|---------|
| **Response Relevance** | 鍥炵瓟涓庨棶棰樼殑鐩稿叧绋嬪害 | LLM-as-Judge锛堜娇鐢?GPT 璇勫垎 1-5锛?|
| **Faithfulness** | 鍥炵瓟鏄惁鍩轰簬妫€绱㈠埌鐨勬枃妗?| 妫€鏌ュ洖绛斾腑鐨勪俊鎭槸鍚︽潵婧愪簬 context |
| **Task Completion Rate** | 鎴愬姛鍥炵瓟闂鐨勬瘮渚?| 姝ｇ‘鍥炵瓟鏁?/ 鎬婚棶棰樻暟 |
| **Response Latency** | 绯荤粺鍝嶅簲鏃堕棿 | 绔埌绔欢杩燂紙绉掞級 |
| **Retrieval Precision** | 妫€绱㈠埌鐨勬枃妗ｇ墖娈电殑鐩稿叧鎬?| 鐩稿叧鐗囨鏁?/ 鎬绘绱㈢墖娈垫暟 |

### 4.3 Ablation Study锛堟秷铻嶅疄楠岋級

鎴戜滑灏嗕粠浠ヤ笅缁村害杩涜娑堣瀺鍒嗘瀽锛?
1. **鍒嗗潡绛栫暐瀵规瘮**锛氫笉鍚?chunk size 瀵规绱㈣川閲忓拰鍥炵瓟鍑嗙‘鎬х殑褰卞搷
2. **妯″瀷瀵规瘮**锛氫笉鍚?LLM 鍚庣鍦ㄧ浉鍚?pipeline 涓嬬殑琛ㄧ幇宸紓
3. **Top-K 妫€绱㈡暟閲?*锛氭绱?1/3/5/10 涓枃妗ｇ墖娈垫椂鐨勬€ц兘鍙樺寲
4. **Embedding 妯″瀷瀵规瘮**锛氫笉鍚?embedding 妯″瀷瀵规绱㈣川閲忕殑褰卞搷

---

## 5. Technical Implementation Plan

### 5.1 Technology Stack

```
Core Framework:    LlamaIndex (Python)
LLM Deployment:    Ollama (local inference)
Embedding Model:   BAAI/bge-small-en-v1.5 or sentence-transformers
Vector Store:      Chroma / FAISS (local)
Frontend (鍙€?:   Gradio / Streamlit
Language:          Python 3.10+
```

### 5.2 Project Structure

```
ScholarLens/
鈹溾攢鈹€ data/
鈹?  鈹溾攢鈹€ papers/              # 瀛︽湳璁烘枃 PDF锛圧AG, Self-RAG, CoT 绛夛級
鈹?  鈹溾攢鈹€ blogs/               # 鎶€鏈崥瀹?URL 鍒楄〃
鈹?  鈹溾攢鈹€ course_materials/    # 璇剧▼璁蹭箟
鈹?  鈹斺攢鈹€ eval/                # 璇勬祴鏁版嵁闆?鈹?      鈹溾攢鈹€ questions.json   # 娴嬭瘯闂 + ground truth
鈹?      鈹斺攢鈹€ README.md        # 鏁版嵁闆嗚鏄?鈹溾攢鈹€ src/
鈹?  鈹溾攢鈹€ indexing/
鈹?  鈹?  鈹溾攢鈹€ data_loader.py   # 澶氭簮鏁版嵁杩炴帴鍣?鈹?  鈹?  鈹溾攢鈹€ chunking.py      # 鍒嗗潡绛栫暐锛?绉嶏級
鈹?  鈹?  鈹斺攢鈹€ metadata.py      # 璁烘枃鍏冩暟鎹彁鍙栵紙鏍囬銆佷綔鑰呫€佸勾浠斤級
鈹?  鈹溾攢鈹€ retrieval/
鈹?  鈹?  鈹斺攢鈹€ retriever.py     # 妫€绱㈠櫒锛堝惈 metadata filtering锛?鈹?  鈹溾攢鈹€ generation/
鈹?  鈹?  鈹溾攢鈹€ llm_config.py    # LLM 鍚庣閰嶇疆
鈹?  鈹?  鈹斺攢鈹€ prompts.py       # Prompt Templates锛堝惈瀵规瘮 prompt锛?鈹?  鈹溾攢鈹€ agent/
鈹?  鈹?  鈹斺攢鈹€ chat_agent.py    # 瀵硅瘽寮?Agent + Memory
鈹?  鈹斺攢鈹€ evaluation/
鈹?      鈹溾攢鈹€ metrics.py        # 璇勪及鎸囨爣瀹炵幇
鈹?      鈹斺攢鈹€ evaluator.py      # 鎵归噺鑷姩璇勪及鍣?鈹溾攢鈹€ notebooks/
鈹?  鈹溾攢鈹€ 01_data_preparation.ipynb   # 鏁版嵁瀵煎叆涓庨澶勭悊
鈹?  鈹溾攢鈹€ 02_single_paper_qa.ipynb    # 鍗曡鏂囬棶绛?Demo
鈹?  鈹溾攢鈹€ 03_cross_paper_compare.ipynb # 璺ㄨ鏂囧姣?Demo
鈹?  鈹溾攢鈹€ 04_chat_agent.ipynb          # 瀵硅瘽 Agent Demo
鈹?  鈹斺攢鈹€ 05_evaluation.ipynb          # 瀹屾暣璇勪及瀹為獙
鈹溾攢鈹€ requirements.txt
鈹斺攢鈹€ README.md
```

---

## 6. Timeline & Milestones

| 闃舵 | 鏃堕棿 | 浠诲姟 | 浜や粯鐗?|
|------|------|------|--------|
| **Phase 1** | Week 1 (Mar 9-15) | 鐜鎼缓 + 鏁版嵁鏀堕泦 | Ollama 閮ㄧ讲瀹屾垚锛屾祴璇曟枃妗ｆ敹闆?|
| **Phase 2** | Week 2 (Mar 16-22) | 鏍稿績 RAG Pipeline 寮€鍙?| Document QA 鍩烘湰鍔熻兘鍙敤 |
| **Phase 3** | Week 3 (Mar 23-25) | 璇勬祴鏁版嵁闆嗘瀯寤?+ 杩涘害鎶ュ憡 | **Progress Report 鎻愪氦 (3/25)** |
| **Phase 4** | Week 4 (Mar 26-Apr 1) | 鍒嗗潡绛栫暐瀹為獙 + 妯″瀷瀵规瘮 | 瀹為獙缁撴灉鍒濇鍒嗘瀽 |
| **Phase 5** | Week 5 (Apr 2-8) | Conversational Agent + 娑堣瀺瀹為獙 | Agent 鍔熻兘瀹屾垚 + 瀹屾暣瀹為獙鏁版嵁 |
| **Phase 6** | Week 6 (Apr 9-14) | 鎶ュ憡鎾板啓 + 婕旂ず鍑嗗 | **Presentation (4/14)** |
| **Final** | Apr 15 | 鏈€缁堟彁浜?| **Final Report + Code (4/15)** |

---

## 7. Division of Labor锛堝垎宸ュ缓璁級

> 浠ヤ笅涓哄缓璁垎宸ユā鏉匡紝鍙牴鎹疄闄呯粍鍛樹汉鏁帮紙1-6浜猴級璋冩暣锛?
| 瑙掕壊 | 涓昏浠诲姟 |
|------|---------|
| **鎴愬憳 A** | 椤圭洰鏋舵瀯璁捐 + RAG Pipeline 鏍稿績寮€鍙?|
| **鎴愬憳 B** | 鏁版嵁鏀堕泦 + 鏁版嵁杩炴帴鍣ㄥ疄鐜?+ 鍒嗗潡绛栫暐瀹為獙 |
| **鎴愬憳 C** | LLM 閮ㄧ讲 (Ollama) + 妯″瀷瀵规瘮瀹為獙 |
| **鎴愬憳 D** | 璇勬祴鏁版嵁闆嗘瀯寤?+ 璇勪及鎸囨爣瀹炵幇 |
| **鎴愬憳 E** | Conversational Agent 寮€鍙?|
| **鎴愬憳 F** | 鎶ュ憡鎾板啓 + 婕旂ず PPT 鍒朵綔 |

---

## 8. Expected Contributions & Innovation

1. **璺ㄨ鏂囧姣旈棶绛?*锛氬尯鍒簬鍗曟枃妗?QA 鐨勫父瑙勫疄鐜帮紝鎴戜滑寮曞叆璺ㄨ鏂囧姣斿姛鑳斤紝閫氳繃 metadata filtering + 瀵规瘮 prompt template 瀹炵幇缁撴瀯鍖栫殑璁烘枃鏂规硶瀵规瘮锛岃繖鏄湰椤圭洰鐨?*鏍稿績鍒涙柊鐐?*
2. **闈㈠悜瀛︽湳鍦烘櫙鐨?RAG 浼樺寲**锛氶拡瀵瑰鏈鏂囩殑鐗圭偣锛堝叕寮忓銆佽〃鏍煎銆佸紩鐢ㄥ叧绯诲鏉傦級锛屾帰绱㈤€傚悎璁烘枃鍦烘櫙鐨勫垎鍧楃瓥鐣?3. **澶氱淮搴﹀垎鍧楃瓥鐣ュ姣?*锛氱郴缁熷寲姣旇緝 4 绉嶅垎鍧楃瓥鐣ュ湪瀛︽湳闂瓟鍦烘櫙涓嬬殑琛ㄧ幇宸紓
4. **杞婚噺绾ч儴缃叉柟妗?*锛氬湪鏈夐檺璁＄畻璧勬簮涓嬶紝閫氳繃閲忓寲妯″瀷瀹炵幇瀹炵敤鐨勫鏈姪鎵?5. **鍙鐜扮殑璇勪及妗嗘灦**锛氭瀯寤哄寘鍚缁村害鎸囨爣鐨勫鏈?QA 璇勬祴鏁版嵁闆嗗拰鑷姩鍖栬瘎浼?pipeline

---

## 9. Risk Assessment & Mitigation

| 椋庨櫓 | 褰卞搷 | 搴斿鎺柦 |
|------|------|---------|
| 璁＄畻璧勬簮涓嶈冻 | 鏃犳硶杩愯澶фā鍨?| 浣跨敤 GPTQ-4bit 閲忓寲 + 灏忔ā鍨?(1.5B-3B) |
| LlamaIndex API 鍙樻洿 | 浠ｇ爜鍏煎鎬ч棶棰?| 鍥哄畾鐗堟湰鍙凤紝鍙傝€冨畼鏂规枃妗?|
| 璇勬祴鏁版嵁闆嗚川閲?| 瀹為獙缁撴灉涓嶅彲闈?| 浜哄伐瀹℃牳 + 浜ゅ弶楠岃瘉 |
| 鏃堕棿绱у紶 | 鍔熻兘涓嶅畬鏁?| 浼樺厛瀹屾垚鏍稿績 QA 鍔熻兘锛孉gent 浣滀负鎵╁睍 |

---

## 10. Demo Scenarios锛堟紨绀哄満鏅璁★級

鍦?4 鏈?14 鏃ョ殑 15 鍒嗛挓婕旂ず涓紝鎴戜滑璁″垝灞曠ず浠ヤ笅鍦烘櫙锛?
**鍦烘櫙 1 - 鍗曡鏂囬棶绛旓紙2 鍒嗛挓锛?*
> 鐢ㄦ埛: "What is the main idea of Self-RAG?"
> ScholarLens: "Self-RAG proposes a framework that learns to retrieve, generate, and critique through self-reflection. Unlike standard RAG which retrieves for every query, Self-RAG trains the LM to adaptively retrieve passages on-demand..." [Source: Asai et al., ICLR 2024]

**鍦烘櫙 2 - 璺ㄨ鏂囧姣旓紙3 鍒嗛挓锛?*
> 鐢ㄦ埛: "Compare the retrieval strategies used in RAG and Self-RAG"
> ScholarLens: "RAG (Lewis et al., 2020) uses a fixed retrieve-then-generate pipeline where retrieval is performed for every input. In contrast, Self-RAG (Asai et al., 2024) introduces reflection tokens that allow the model to decide when to retrieve..."

**鍦烘櫙 3 - 澶氳疆瀵硅瘽锛? 鍒嗛挓锛?*
> 鐢ㄦ埛: "Tell me about Chain-of-Thought prompting"
> ScholarLens: [鍥炵瓟 CoT 姒傝堪]
> 鐢ㄦ埛: "How does it compare with Self-Consistency?"
> ScholarLens: [鍩轰簬涓婁笅鏂囪繘琛屽姣擼

**鍦烘櫙 4 - 瀹為獙缁撴灉灞曠ず锛? 鍒嗛挓锛?*
灞曠ず涓嶅悓鍒嗗潡绛栫暐銆佷笉鍚屾ā鍨嬨€佷笉鍚?Top-K 涓嬬殑鎬ц兘瀵规瘮鍥捐〃銆?
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
