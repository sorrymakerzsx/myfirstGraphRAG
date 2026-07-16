# 三国演义知识图谱 RAG 系统

基于 GraphRAG 思想构建的《三国演义》知识图谱问答系统，融合混合检索、图扩展、社区发现与 LLM 生成，支持 CLI 交互与 Web API 两种调用方式。

## 核心特性

- **知识图谱**：82 节点 / 150 边，覆盖 57 位人物、4 大势力、21 个历史事件
- **混合检索**：BM25 稀疏检索 + 向量稠密检索，RRF 融合排序
- **Cross-Encoder 精排**：BAAI/bge-reranker-base 联合编码 query+doc
- **图扩展检索**：2 跳 BFS 多跳推理 + 子图关系提取 + 关系推理链
- **GraphRAG 分层检索**：社区发现 + 社区摘要向量匹配
- **查询理解**：实体链接（别名→标准名）、查询扩展（同义词→关系词）、意图分类
- **LLM 生成**：DeepSeek 双角色 Prompt，无 key 时回退模板模式
- **增量更新**：CLI/API 增删实体关系，自动重建索引
- **Web API**：FastAPI 封装，9 个 REST 接口，自带 Swagger 文档

## 项目结构

```
.
├── main.py                 # CLI 交互入口
├── api.py                  # FastAPI Web API 服务
├── kg_builder.py           # 知识图谱构建（NetworkX）
├── rag_engine.py           # RAG 检索引擎核心
├── query_understanding.py  # 查询理解（实体链接/扩展/意图分类）
├── entity_extractor.py     # 实体关系抽取
├── evaluation.py           # 评估模块（Recall/MRR/NDCG）
├── kg_neo4j.py             # Neo4j 存储版本（可选）
├── requirements.txt        # 依赖列表
├── data/
│   ├── three_kingdoms.json # 知识图谱数据
│   └── eval_dataset.json   # 评估数据集（22 用例）
└── RAG进阶步骤和建议.md     # 开发笔记
```

> 模型文件（embedding_model / reranker_model）首次运行时自动从 HuggingFace 下载，已通过 `.gitignore` 排除。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 LLM（可选）

启用 LLM 自然语言回答，设置环境变量即可：

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "your-api-key"

# Linux/macOS
export DEEPSEEK_API_KEY=your-api-key
```

未配置时系统自动回退到模板回答模式，不影响检索功能。

### 3. 启动 CLI 交互

```bash
python main.py
```

进入交互式终端后可用命令：

```
search 关羽              # 搜索实体
neighbors guanyu         # 查看关系网络
ask 关羽的敌人是谁        # RAG 智能问答
ask 关羽怎么死的          # 被动意图查询
community                # 查看社区发现结果
eval                     # 运行评估（22 用例）
add_entity character simayi 司马懿 曹魏 字仲达
add_relation guanyu lvbu 敌对 虎牢关三英战吕布
serve 8000               # 启动 Web API
```

### 4. 启动 Web API

```bash
python api.py --port 8000
# 或在 CLI 中执行：serve 8000
```

访问 `http://localhost:8000/docs` 查看自动生成的 Swagger 文档。

主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/stats` | 图谱统计 |
| GET | `/search?name=关羽` | 搜索实体 |
| GET | `/entity/{id}` | 实体详情 |
| GET | `/entity/{id}/neighbors` | 实体邻居 |
| POST | `/ask` | RAG 智能问答 |
| POST | `/entity` | 添加实体 |
| POST | `/relation` | 添加关系 |
| DELETE | `/entity/{id}` | 删除实体 |
| DELETE | `/relation` | 删除关系 |

## 检索流程

```
用户问题
   │
   ▼
查询理解（实体链接 + 查询扩展 + 意图分类）
   │
   ▼
混合召回（BM25 + 向量检索 + RRF 融合）
   │  + 实体注入 + 1跳邻居注入（带方向记录）
   ▼
Cross-Encoder 精排
   │  + 实体加分 + 关系加分 + 方向感知
   ▼
三路融合上下文构建
   ├── 多跳图扩展（2跳BFS带路径）
   ├── 子图关系提取
   ├── 社区检索（GraphRAG分层）
   └── 关系推理链
   │
   ▼
LLM 生成（DeepSeek 双角色 Prompt）
   │  无 key → 回退模板模式
   ▼
答案 + 完整检索过程
```

## 关键策略说明

### 混合检索 + RRF 融合
BM25 负责关键词精确匹配（人名、地名），向量检索负责语义匹配（同义词、近义词）。两者分数尺度不同，用 Reciprocal Rank Fusion（`score = Σ 1/(60+rank)`）只按排名融合，免调参。

### Cross-Encoder 精排
召回阶段用双塔模型（query 和 doc 独立编码，快但粗），精排阶段用 Cross-Encoder（query+doc 联合编码，慢但精）。两阶段架构兼顾速度和精度。

### 邻居注入
查询中明确提到的实体，注入其 1 跳邻居到召回候选集。解决"答案实体未召回"问题：如"关羽怎么死的"答案吕蒙是关羽的邻居，但吕蒙的描述中可能没有查询关键词。

### 方向感知
查询"关羽怎么死的"是被动意图，关羽是受事者。重排序时只匹配入边（吕蒙→关羽 擒杀），不匹配出边（关羽→文丑 斩杀），避免混淆施事/受事。

### GraphRAG 分层检索
用 NetworkX 的 `greedy_modularity_communities` 做社区发现，为每个社区生成摘要。检索时先匹配社区摘要（宏观），再匹配具体实体（微观）。

## 评估指标

基于 22 个测试用例的评估结果：

| 指标 | 值 | 说明 |
|------|-----|------|
| Recall@3 | 0.8894 | 前 3 结果含正确答案的比例 |
| Recall@5 | 0.9136 | 前 5 结果含正确答案的比例 |
| MRR | 0.5909 | 正确答案的平均排名倒数 |
| NDCG@5 | 0.6760 | 考虑排名位置的归一化增益 |
| Answer Accuracy | 0.7409 | LLM 答案正确率 |

运行评估：

```bash
python main.py
# 在 CLI 中输入：eval
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 图谱存储 | NetworkX DiGraph + JSON |
| 嵌入模型 | BAAI/bge-base-zh-v1.5（768 维） |
| 重排序模型 | BAAI/bge-reranker-base |
| 稀疏检索 | rank_bm25 |
| 社区发现 | greedy_modularity_communities |
| LLM | DeepSeek（OpenAI 兼容接口） |
| Web 框架 | FastAPI + Uvicorn |
| CLI 渲染 | Rich |

## 数据格式

知识图谱数据存储在 `data/three_kingdoms.json`，格式如下：

```json
{
  "entities": {
    "characters": [
      {"id": "guanyu", "name": "关羽", "faction": "蜀汉", "description": "字云长..."}
    ],
    "factions": [...],
    "events": [...]
  },
  "relationships": [
    {"source": "guanyu", "target": "zhangfei", "relation": "结义兄弟", "detail": "桃园结义"}
  ]
}
```

支持通过 CLI 或 API 增删实体和关系，修改后自动保存并重建索引。

## License

MIT
