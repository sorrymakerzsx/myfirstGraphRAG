## 一、修改了什么，在哪里改的

### 1. 丰富知识库数据

**文件**: [data/three_kingdoms.json](file:///d:/agent_learning/quicktryGraphRag/data/three_kingdoms.json)

**修改内容**:
- 人物从 22 个增加到 **57 个**（新增庞统、姜维、魏延、马谡、徐庶、法正、曹丕、曹植、张辽、徐晃、张郃、荀彧、司马昭、庞德、孙坚、孙策、黄盖、甘宁、太史慈、诸葛瑾、大乔、小乔、韩当、华雄、颜良、文丑、韩玄、刘表、丁原、公孙瓒、刘禅、严颜、马岱、夏侯渊等）
- 事件从 7 个增加到 **21 个**（新增温酒斩华雄、三英战吕布、过五关斩六将、草船借箭、空城计、挥泪斩马谡、白帝城托孤、七擒孟获、连环计、煮酒论英雄、战长沙、定军山之战、逍遥津之战、九伐中原等）
- 关系边从 44 条增加到 **149 条**

**关于黄忠**：黄忠最初是**刘表**的部下（不是韩当，韩当是东吴将领），刘表死后归属长沙太守**韩玄**。关羽战长沙时，韩玄怀疑黄忠通敌要杀他，**魏延**杀韩玄救黄忠，二人一同归降刘备。

**你自己怎么加数据**：直接编辑 `data/three_kingdoms.json`，按格式在 `entities.characters` 数组中加人物，在 `relationships` 数组中加关系即可。格式：
```json
{"source": "人物ID", "target": "目标ID", "relation": "关系类型", "detail": "详细说明"}
```

### 2. 修复颜色对比度问题

**文件**: [kg_builder.py](file:///d:/agent_learning/quicktryGraphRag/kg_builder.py)

**修改了三处**:

| 位置                        | 问题                     | 修复方式                                                     |
| --------------------------- | ------------------------ | ------------------------------------------------------------ |
| HTML 节点字体 (第300行)     | 白色字在浅色节点上看不清 | 每个节点加 `font: {color: "#ffffff", strokeWidth: 3, strokeColor: "#000000"}`，白色文字+黑色描边 |
| HTML 边标签 (第370行)       | 灰色字在深色背景上看不清 | 改为 `font: {color: "#ffffff", strokeWidth: 3, strokeColor: "#000000"}`，白色文字+黑色描边 |
| Matplotlib 边标签 (第250行) | 文字与节点/边重叠看不清  | 加 `bbox=dict(facecolor="white", alpha=0.7)` 白色背景框      |

**你自己怎么调颜色**：在 `kg_builder.py` 第157行的 `FACTION_COLORS` 字典中修改：
```python
FACTION_COLORS = {
    "蜀汉": "#FF6B6B",   # 改这里
    "曹魏": "#4ECDC4",
    "东吴": "#45B7D1",
    "群雄": "#96CEB4",
}
```

---

## 二、当前 RAG 用到的策略和原因

### 策略 1：TF-IDF 字符级 n-gram 向量检索

**位置**: [rag_engine.py](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py) 第23-28行

```python
TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
```

**为什么用这个**:
- 中文不需要分词，字符级 n-gram 天然支持中文语义匹配
- 2-4 gram 能捕获"诸葛亮"→"诸葛"、"黄忠"→"忠"等部分匹配
- 无需下载大模型，秒级启动，适合快速原型

**局限**: 无法理解同义词（"死" vs "阵亡"），后续可换 sentence-transformers 的多语言模型。

### 策略 2：知识图谱文档化

**位置**: rag_engine.py 第30-58行 `_build_documents()`

将每个图谱节点转换为一段富文本：**实体描述 + 所有关系拼接**。这样向量检索不仅匹配实体名，还能匹配关系描述中的关键词。

### 策略 3：图扩展检索（Graph Expand）

**位置**: rag_engine.py 第87-106行 `_graph_expand()`

**流程**:
```
用户问题 → TF-IDF 向量检索 → 命中 Top-K 实体 → 沿图谱边扩展1跳邻居
```

**为什么需要图扩展**: 向量检索只能找到直接匹配的实体，但用户问"关羽怎么死的"，不仅需要关羽的信息，还需要吕蒙（擒杀者）、陆逊（间接相关）的信息。图扩展把这些关联实体一并召回。

### 策略 4：模板化回答生成

**位置**: rag_engine.py 第149-175行 `answer()`

当前没有接 LLM，用模板组织回答：主实体信息 + 关联实体列表。这是 GraphRAG 的基础形态。

---

## 三、后续如何进一步开发以掌握 RAG 和 GraphRAG

### 阶段一：增强检索（掌握 RAG 基础）

| 改进点               | 怎么做                                                       | 学到什么                     |
| -------------------- | ------------------------------------------------------------ | ---------------------------- |
| 换更好的嵌入模型     | 在 `rag_engine.py` 中用 `sentence-transformers` 的 `paraphrase-multilingual-MiniLM-L12-v2` 替换 TF-IDF | 语义嵌入 vs 词频统计的区别   |
| 加重排序（Reranker） | 召回后用 cross-encoder 对 Top-K 结果重排序                   | 召回率 vs 精确率的权衡       |
| 混合检索             | BM25 + 向量检索分数融合                                      | 稀疏检索 vs 稠密检索的互补性 |
| 接入 LLM 生成回答    | 把 `answer()` 中的模板替换为调用 OpenAI/本地 LLM，传入检索上下文 | Prompt Engineering           |

**具体操作**: 在 `rag_engine.py` 的 `answer()` 方法中，把 `context` 传给 LLM：
```python
# 伪代码
prompt = f"基于以下知识图谱信息回答问题：\n{result['context']}\n\n问题：{question}"
answer = llm.chat(prompt)
```

### 阶段二：深化 GraphRAG（掌握图检索）

| 改进点       | 怎么做                                     | 学到什么                |
| ------------ | ------------------------------------------ | ----------------------- |
| 多跳推理     | 把 `_graph_expand()` 的深度从 1 改为 2-3   | 图遍历算法（BFS/DFS）   |
| 社区发现     | 用 NetworkX 的 `community` 模块对图谱分群  | 图算法在 RAG 中的应用   |
| 子图检索     | 查询时提取相关子图作为上下文，而非单个实体 | 子图同构、图注意力      |
| 关系推理     | 查询"关羽的敌人的主公是谁"这类多跳问题     | 知识图谱推理            |
| 图数据库迁移 | 从 NetworkX 迁移到 Neo4j                   | 图数据库查询语言 Cypher |

**具体操作** — 多跳扩展示例，改 `rag_engine.py`:
```python
def _graph_expand(self, entity_ids, max_depth=2):  # 改为2跳
    visited = set(entity_ids)
    expanded = []
    frontier = entity_ids
    for depth in range(max_depth):
        next_frontier = []
        for entity_id in frontier:
            for n in self.kg.get_neighbors(entity_id):
                nid = n["entity"].get("id", "")
                if nid and nid not in visited:
                    visited.add(nid)
                    next_frontier.append(nid)
                    expanded.append(...)
        frontier = next_frontier
    return expanded
```

### 阶段三：生产级 GraphRAG（进阶）

1. **实体抽取 + 自动建图**: 用 NLP 从原文自动抽取实体和关系，而非手工编辑 JSON
2. **GraphRAG 论文复现**: 微软 GraphRAG 论文的社区摘要 + 分层检索
3. **向量+图+LLM 三路融合**: 向量检索负责语义匹配，图检索负责关系推理，LLM 负责生成

### 关键文件速查

| 文件                                                         | 作用            | 你需要改的地方       |
| ------------------------------------------------------------ | --------------- | -------------------- |
| [data/three_kingdoms.json](file:///d:/agent_learning/quicktryGraphRag/data/three_kingdoms.json) | 知识图谱数据    | 加人物/事件/关系     |
| [kg_builder.py](file:///d:/agent_learning/quicktryGraphRag/kg_builder.py) | 图谱构建+可视化 | 改颜色/布局/加图算法 |
| [rag_engine.py](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py) | RAG 检索引擎    | 改检索策略/接 LLM    |
| [main.py](file:///d:/agent_learning/quicktryGraphRag/main.py) | CLI 交互        | 加新命令             |

## 可继续提升的方向（按优先级排序）
### 1. 修复 Cross-Encoder 重排序（当前回退模式）
问题 : data/reranker_model/ 下载损坏，一直在用语义相似度回退

怎么改 : 删除损坏目录，换用中文专用重排序模型 BAAI/bge-reranker-base

学到什么 : 中文 RAG 为什么需要中文专用模型而非多语言通用模型

### 2. 查询理解与改写
当前 : 直接用用户原文做检索，"关二爷的对手" 检索不到

可做 :

- 实体链接：关二爷 → 关羽、卧龙 → 诸葛亮
- 查询意图分类：单跳问题 vs 多跳问题 vs 比较问题
- 查询扩展："对手" → "敌对 + 交锋 + 曾为敌手"
学到什么 : Query Understanding 是 RAG 的入口环节

### 3. 评估体系
当前 : 没有量化指标，全靠人眼判断

可做 : 构建 20-30 个标准 Q&A 对，计算 Recall@K、MRR、NDCG

学到什么 : RAG 系统怎么量化评估检索质量

### 4. 接入 LLM 生成自然语言回答
当前 : 模板模式，回答是拼接的

可做 : 设置 OPENAI_API_KEY ，让 LLM 基于检索上下文生成流畅回答

学到什么 : Prompt Engineering、上下文窗口管理

### 5. 微软 GraphRAG 完整复现
当前 : 只实现了社区摘要，没有社区报告和分层社区

可做 :

- 多层级社区（粗→细分层聚类）
- 每个社区生成详细报告（LLM 总结）
- 全局搜索 vs 局部搜索两种模式
学到什么 : 微软 GraphRAG 论文的核心架构

### 6. 中文专用嵌入模型
当前 : paraphrase-multilingual-MiniLM-L12-v2 （多语言通用）

可做 : 换成 BAAI/bge-large-zh-v1.5 或 moka-ai/m3e-large （中文专用）

学到什么 : 领域适配的嵌入模型 vs 通用模型的效果差异

### 7. 图谱增量更新
当前 : 每次改数据要手动编辑 JSON 再重启

可做 : CLI 命令直接增删实体和关系，自动重建索引

学到什么 : 增量索引 vs 全量重建的权衡

### 8. Web API 服务化
当前 : CLI 交互

可做 : 用 FastAPI 包装成 HTTP API，支持前端调用

学到什么 : RAG 系统的工程化部署


下面是完整的项目流程与策略梳理。

---

# 三国演义知识图谱 RAG 系统 - 全流程与策略梳理

## 一、整体架构

```
用户问题
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ 1. 查询理解 (query_understanding.py)                     │
│    实体链接 → 查询扩展 → 意图分类                          │
└──────────────────────────────────────────────────────────┘
   │ 改写后的查询 + 扩展关系词 + 链接实体
   ▼
┌──────────────────────────────────────────────────────────┐
│ 2. 混合召回 (rag_engine.py _hybrid_search)                │
│    BM25 稀疏检索 + 向量稠密检索 → RRF 融合                 │
│    + 实体注入 + 1跳邻居注入（带方向记录）                   │
└──────────────────────────────────────────────────────────┘
   │ Top-N 候选
   ▼
┌──────────────────────────────────────────────────────────┐
│ 3. 重排序 (_rerank)                                       │
│    Cross-Encoder 精排 + 实体加分 + 关系加分 + 方向感知      │
└──────────────────────────────────────────────────────────┘
   │ Top-5 精选实体
   ▼
┌──────────────────────────────────────────────────────────┐
│ 4. 三路融合上下文构建                                      │
│    ① 多跳图扩展（2跳BFS带路径）                            │
│    ② 子图检索（命中实体间内部边）                          │
│    ③ 社区检索（GraphRAG分层 + 命中实体所在社区）           │
│    ④ 关系推理链（多跳关系链查询）                          │
└──────────────────────────────────────────────────────────┘
   │ 完整上下文
   ▼
┌──────────────────────────────────────────────────────────┐
│ 5. 答案生成                                               │
│    LLM 双角色 Prompt（DeepSeek）→ 无 key 回退模板模式      │
└──────────────────────────────────────────────────────────┘
   │
   ▼
  答案 + 检索过程（CLI 展示 / API 返回 JSON）
```

---

## 二、各阶段策略与选择理由

### 阶段 0：知识图谱构建（[kg_builder.py](file:///d:/agent_learning/quicktryGraphRag/kg_builder.py)）

| 策略                 | 实现                                                         | 选择理由                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **NetworkX DiGraph** | 有向图，节点带 type/name/faction/description 属性            | 关系有方向（如"斩杀"是单向的），DiGraph 能区分出边（施事）和入边（受事） |
| **JSON 文件存储**    | [data/three_kingdoms.json](file:///d:/agent_learning/quicktryGraphRag/data/three_kingdoms.json) | 82节点规模小，JSON 可读性强，便于手工校对和增量编辑          |
| **文档化**           | [rag_engine.py#L188-213](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L188-L213) `_build_documents()` 将每个节点+邻居关系拼成一段文本 | 检索需要文本形式，把图结构"拍平"为文档，让 BM25 和向量检索都能用 |
| **社区发现**         | `greedy_modularity_communities`（模块度最大化）              | 自动发现人物群落（如"蜀汉五虎将"会聚到一个社区），为 GraphRAG 分层检索提供高层视角 |

---

### 阶段 1：查询理解（[query_understanding.py](file:///d:/agent_learning/quicktryGraphRag/query_understanding.py)）

| 策略             | 实现                                                         | 选择理由                                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **实体链接**     | [L131-153](file:///d:/agent_learning/quicktryGraphRag/query_understanding.py#L131-L153) 别名表 + 按长度降序匹配 | 三国有大量别称（关二爷→关羽、卧龙→诸葛亮），不链接会导致检索召回失败。长度降序避免"关公"被"公"误匹配 |
| **别名动态加载** | [L108-129](file:///d:/agent_learning/quicktryGraphRag/query_understanding.py#L108-L129) 从 description 中正则提取"字XX""号XX" | 静态表覆盖不全，动态从数据里提取"字云长"→关羽，增量更新后自动生效 |
| **查询扩展**     | [L63-94](file:///d:/agent_learning/quicktryGraphRag/query_understanding.py#L63-L94) 同义词→关系类型 | "对手"→敌对+曾为敌手，"死"→斩杀+擒杀+弑杀。因为图谱里的 relation 是标准化的，用户口语词汇需要映射 |
| **意图分类**     | [L97-101](file:///d:/agent_learning/quicktryGraphRag/query_understanding.py#L97-L101) 正则匹配 multi_hop/comparison/description | "X的Y的Z"是多跳，"X和Y谁强"是比较，分类后可指导后续策略（目前用于诊断，未直接驱动检索） |

---

### 阶段 2：混合召回（[rag_engine.py#L318-349](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L318-L349)）

| 策略              | 实现                                                         | 选择理由                                                     |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **BM25 稀疏检索** | [L282-299](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L282-L299) 字符级分词 | 中文无空格，BM25 默认按词分词需 jieba；字符级简单且对人名（"关羽"="关"+"羽"）有效 |
| **向量稠密检索**  | [L301-316](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L301-L316) BAAI/bge-base-zh-v1.5（768维） | 中文专用模型，能捕获"卧龙"和"诸葛亮"的语义相似度，弥补 BM25 无法处理同义词的缺陷 |
| **RRF 融合**      | [L337-345](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L337-L345) `score = Σ 1/(60 + rank)` | BM25 和向量分数尺度不同，直接加权需调参；RRF 只用排名，免调参且效果稳健（业界标准做法） |
| **实体注入**      | [L556-574](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L556-L574) 查询中明确提到的实体强制入候选 | 避免"关羽的敌人"这类查询中，关羽因 BM25/向量分数低而漏召回，导致后续推理无起点 |
| **1跳邻居注入**   | [L576-592](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L576-L592) 注入查询实体的直接邻居 + 记录 (relation, direction) | 解决"答案实体未召回"问题：吕蒙擒杀关羽，但吕蒙的描述里可能没有"关羽怎么死"的关键词，只有作为关羽邻居才能被找到 |

**邻居注入的核心价值**（项目最大提升点）：Recall@5 从 0.5394 → 0.7667

---

### 阶段 3：重排序（[rag_engine.py#L351-411](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L351-L411)）

| 策略                   | 实现                                                         | 选择理由                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Cross-Encoder 精排** | [L370-374](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L370-L374) BAAI/bge-reranker-base 联合编码 query+doc | 召回阶段双塔模型（query 和 doc 独立编码）精度有限；Cross-Encoder 把 query 和 doc 拼一起做注意力，能捕获细粒度匹配。Recall@5 从 0.7667 → 0.9136 |
| **回退机制**           | [L375-380](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L375-L380) Cross-Encoder 失败→余弦相似度 | 模型下载可能因网络失败，保证系统可用性                       |
| **实体加分 +1.0**      | [L392-394](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L392-L394) 实体名出现在查询中 | 之前"关羽的敌人的主公"查询中，关羽排名第4被荀彧挤掉，加1.0后关羽排第1，推理链才能从关羽出发 |
| **强实体加分 +2.5**    | [L395-397](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L395-L397) 查询理解明确链接到的实体 | 通过别名链接到的实体（关二爷→关羽）比单纯文本匹配更确定，给更大权重 |
| **关系加分 +2.0**      | [L399-408](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L399-L408) 邻居关系匹配查询扩展词 | "关羽的敌人"查询中，吕蒙与关羽的关系是"敌对"，匹配扩展词，加分后吕蒙排名上升 |
| **方向感知**           | [L403-404](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L403-L404) 被动意图只匹配入边 | "关羽怎么死的"应匹配"吕蒙→关羽(擒杀)"入边，而非"关羽→文丑(斩杀)"出边。避免混淆施事/受事 |

---

### 阶段 4：三路融合上下文构建

| 策略                 | 实现                                                         | 选择理由                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **多跳图扩展**       | [L413-429](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L413-L429) 2跳BFS，记录完整路径 | "X的敌人的主公"需要2跳推理，BFS天然支持多跳，路径记录让 LLM 能解释推理过程 |
| **子图检索**         | [L452-454](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L452-L454) 提取命中实体集的内部边 | "诸葛亮和司马懿什么关系"这类对比查询，需要两个实体间的直接关系，子图正好提供 |
| **社区检索（分层）** | [L260-278](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L260-L278) 社区摘要做向量匹配 | 微软 GraphRAG 核心思想：先找相关社区（宏观），再找具体实体（微观）。对于"蜀汉有哪些猛将"这类宽泛问题，社区级比实体级更合适 |
| **关系推理链**       | [L456-526](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L456-L526) 关系关键词映射 + 多跳链查询 | 专门解决多跳推理问题：从"敌对""主公"等关系词出发，沿图谱关系链查找，比纯向量检索更精确 |

---

### 阶段 5：答案生成（[rag_engine.py#L738-770](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L738-L770)）

| 策略                  | 实现                                                         | 选择理由                                                     |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **LLM 双角色 Prompt** | [L740-756](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L740-L756) system 放规则，user 放上下文+问题 | system 角色固定约束模型行为（不编造、无数据说不知道），user 角色放可变内容。分离后规则不会被上下文污染 |
| **严格规则约束**      | [L743-744](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L743-L744) "只基于检索上下文""无数据直接说不知道" | 符合项目约束：图谱没记录就明确说不知道，不输出无关实体。避免 LLM 用训练知识"幻觉" |
| **temperature=0.3**   | [L766](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L766) | 低温度保证回答稳定可控，知识问答不需要创造性                 |
| **模板回退**          | [L772-](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L772-) `_template_answer()` | 无 API key 或调用失败时仍可使用，保证系统可用性              |

---

### 阶段 6：增量更新（[kg_builder.py#L154-243](file:///d:/agent_learning/quicktryGraphRag/kg_builder.py#L154-L243)）

| 策略               | 实现                                                         | 选择理由                                                     |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **全量重建索引**   | [rag_engine.py#L152-186](file:///d:/agent_learning/quicktryGraphRag/rag_engine.py#L152-L186) `rebuild_index()` | 82节点小图谱全量重建约15秒；BM25 和社区发现不支持增量，全量重建比维护增量逻辑更简单可靠 |
| **模型不重复加载** | rebuild_index 只重建文档/向量/BM25，不重新加载模型           | 嵌入模型和 Cross-Encoder 加载耗时（数秒），且与数据无关，复用即可 |

---

### 阶段 7：Web API 服务化（[api.py](file:///d:/agent_learning/quicktryGraphRag/api.py)）

| 策略                      | 实现                                                         | 选择理由                                                     |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **lifespan 上下文管理器** | [api.py#L29-47](file:///d:/agent_learning/quicktryGraphRag/api.py#L29-L47) | FastAPI 推荐方式，替代 `@app.on_event("startup")`。启动时初始化一次，所有请求复用，避免每个请求重新加载模型（15秒） |
| **全局引擎复用**          | `_kg` / `_rag` 全局变量                                      | 模型加载一次，后续请求毫秒级响应                             |
| **Pydantic 校验**         | [api.py#L60-83](file:///d:/agent_learning/quicktryGraphRag/api.py#L60-L83) | 自动 JSON 解析 + 类型校验 + 友好的 422 错误，省去手写校验代码 |
| **写操作自动重建**        | POST/DELETE 接口调用 `rebuild_index()`                       | API 用户无需关心索引同步，添加实体后立即可查询               |

---

### 阶段 8：评估体系（[evaluation.py](file:///d:/agent_learning/quicktryGraphRag/evaluation.py)）

| 指标            | 含义                    | 最终值 |
| --------------- | ----------------------- | ------ |
| Recall@3        | 前3个结果是否含正确答案 | 0.8894 |
| Recall@5        | 前5个结果是否含正确答案 | 0.9136 |
| MRR             | 正确答案的平均排名倒数  | 0.5909 |
| NDCG@5          | 考虑排名位置的增益      | 0.6760 |
| Answer Accuracy | LLM/模板答案的正确率    | 0.7409 |

---

## 三、关键演进数据

| 优化阶段            | Recall@5 | MRR    | 核心改动                                        |
| ------------------- | -------- | ------ | ----------------------------------------------- |
| 初始版（BM25+向量） | 0.5394   | 0.3977 | 基础混合检索                                    |
| +邻居注入           | 0.7667   | 0.5167 | 解决答案实体未召回                              |
| +Cross-Encoder      | 0.9136   | 0.5909 | 精排提升排名质量                                |
| +中文嵌入模型       | 0.9136   | 0.5909 | 效果不明显（BM25+CrossEncoder双重保障掩盖差异） |

**关键发现**：在 BM25+向量混合召回 + Cross-Encoder 精排的架构下，嵌入模型的升级对最终效果影响很小（+0.0007 NDCG@5），因为 BM25 保障了关键词匹配，Cross-Encoder 保障了精排质量，嵌入模型的语义匹配作用被双重保障掩盖。

---

## 四、剩余可优化方向

1. **#9 multi_hop 失败案例**："关羽的敌人的主公是谁"——需2跳推理链注入，当前邻居注入只做1跳
2. **意图分类驱动检索策略**：当前意图分类仅用于诊断，未直接改变检索路径
3. **增量索引**：当前增量更新走全量重建，大规模图谱需真正的增量索引
4. **事件参与者注入**：事件类查询（如"赤壁之战参与者"）召回率待提升

如需深入某个阶段的实现细节或继续优化某个方向，告诉我即可。