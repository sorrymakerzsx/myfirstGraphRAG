"""
RAG 检索引擎 v2 - 阶段一增强版
包含 4 项改进：
  1. 语义嵌入模型（sentence-transformers 多语言模型）替代 TF-IDF
  2. BM25 + 向量混合检索（Reciprocal Rank Fusion 分数融合）
  3. Cross-Encoder 重排序（精确排序 Top-K）
  4. LLM 生成回答（可配置，无 API key 时回退到增强模板）
"""
import os
import sys
import shutil
from pathlib import Path

# 确保 libs 目录中的依赖可用（dill 等）
# 用 append 而非 insert(0, ...) 避免覆盖环境自带的包（如 numpy）
_libs_path = str(Path(__file__).parent / "libs")
if _libs_path not in sys.path:
    sys.path.append(_libs_path)

# Windows 下 HuggingFace 创建 symlink 需要管理员权限，改用文件复制
if sys.platform == "win32":
    _original_symlink = os.symlink
    def _safe_symlink(src, dst, *args, **kwargs):
        try:
            _original_symlink(src, dst, *args, **kwargs)
        except OSError:
            # HF 用相对路径创建 symlink，copy 需要先转绝对路径
            if not os.path.isabs(src):
                src = os.path.normpath(os.path.join(os.path.dirname(dst), src))
            shutil.copy2(src, dst)
    os.symlink = _safe_symlink

# 使用国内 HuggingFace 镜像，避免 CDN 超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import numpy as np
from rank_bm25 import BM25Okapi

from kg_builder import KnowledgeGraph
from query_understanding import QueryUnderstanding


class RAGEngine:
    """知识图谱 RAG 引擎 v4：查询理解 + 混合检索 + 重排序 + 多跳图 + 社区摘要 + 三路融合 + LLM"""

    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding_model: str = "BAAI/bge-base-zh-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        llm_config: dict = None,
    ):
        self.kg = kg
        self.llm_config = llm_config or {}

        # 查询理解模块（阶段三增强）
        self.query_understanding = QueryUnderstanding(
            str(Path(__file__).parent / "data" / "three_kingdoms.json")
        )

        # --- 文档构建 ---
        self.docs = self._build_documents()
        self.doc_texts = [d["text"] for d in self.docs]
        self.doc_ids = [d["id"] for d in self.docs]

        # --- 0. 社区发现 + 社区摘要（阶段二+三） ---
        print("[0/6] 社区发现...")
        self.community_info = self.kg.detect_communities()
        print(f"  检测到 {len(self.community_info['communities'])} 个社区")

        print("[1/6] 生成社区摘要（GraphRAG 分层检索）...")
        self.community_summaries = self._build_community_summaries()
        self.community_docs = [s["text"] for s in self.community_summaries]
        print(f"  生成 {len(self.community_summaries)} 个社区摘要")

        # --- 2. 语义嵌入模型 ---
        print(f"[2/6] 加载语义嵌入模型: {embedding_model}")
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import snapshot_download

        model_local_path = str(Path(__file__).parent / "data" / "embedding_model")
        reranker_local_path = str(Path(__file__).parent / "data" / "reranker_model")

        # 检查模型文件是否完整（config.json 是必需文件）
        # 支持完整 repo_id（如 BAAI/bge-base-zh-v1.5），不再写死 sentence-transformers 前缀
        # snapshot_download 自带断点续传：已下载的文件会跳过，只下载缺失文件
        need_download = True
        if Path(model_local_path).exists():
            if (Path(model_local_path) / "config.json").exists():
                need_download = False

        if need_download:
            print(f"  下载嵌入模型到 {model_local_path} ...")
            snapshot_download(
                repo_id=embedding_model,
                local_dir=model_local_path,
            )
        self.encoder = SentenceTransformer(model_local_path)
        self.doc_embeddings = self.encoder.encode(
            self.doc_texts, show_progress_bar=True, convert_to_numpy=True
        )
        # 社区摘要也做嵌入
        self.community_embeddings = self.encoder.encode(
            self.community_docs, show_progress_bar=False, convert_to_numpy=True
        )
        print(f"  嵌入维度: {self.doc_embeddings.shape[1]}")

        # --- 3. BM25 稀疏检索 ---
        print("[3/6] 构建 BM25 索引...")
        self.doc_tokens = [list(text) for text in self.doc_texts]
        self.bm25 = BM25Okapi(self.doc_tokens)
        print(f"  BM25 索引完成，{len(self.doc_tokens)} 篇文档")

        # --- 4. 重排序器 ---
        self.reranker = None
        self.reranker_type = "none"
        print(f"[4/6] 加载重排序模型: {reranker_model}")
        try:
            from sentence_transformers import CrossEncoder

            # 检查模型文件是否完整（config.json 是必需文件）
            # snapshot_download 自带断点续传，已下载文件会跳过
            need_download = True
            if Path(reranker_local_path).exists():
                if (Path(reranker_local_path) / "config.json").exists():
                    need_download = False

            if need_download:
                print(f"  下载重排序模型到 {reranker_local_path} ...")
                snapshot_download(
                    repo_id=reranker_model,
                    local_dir=reranker_local_path,
                )
            self.reranker = CrossEncoder(reranker_local_path)
            self.reranker_type = "cross-encoder"
            print("  Cross-Encoder 重排序器就绪")
        except Exception as e:
            print(f"  Cross-Encoder 加载失败: {e}")
            print("  回退到语义相似度重排序（基于嵌入模型余弦相似度）")
            self.reranker_type = "semantic"

        # --- 5. 多跳图索引（阶段二） ---
        print("[5/6] 构建多跳图索引...")
        self.max_hop_depth = 2
        print(f"  最大跳数: {self.max_hop_depth}")

        # --- 6. LLM 配置 ---
        print("[6/6] 初始化 LLM...")
        self.llm = self._init_llm()

    def rebuild_index(self):
        """
        增量更新后重建索引（全量重建）
        重建：文档 → 社区 → 向量嵌入 → BM25
        不重建：嵌入模型、重排序模型、LLM（模型只需加载一次）
        """
        print("[重建] 1/5 重建文档...")
        self.docs = self._build_documents()
        self.doc_texts = [d["text"] for d in self.docs]
        self.doc_ids = [d["id"] for d in self.docs]

        print("[重建] 2/5 重建社区发现...")
        self.community_info = self.kg.detect_communities()
        self.community_summaries = self._build_community_summaries()
        self.community_docs = [s["text"] for s in self.community_summaries]

        print("[重建] 3/5 重新编码文档向量...")
        self.doc_embeddings = self.encoder.encode(
            self.doc_texts, show_progress_bar=False, convert_to_numpy=True
        )
        self.community_embeddings = self.encoder.encode(
            self.community_docs, show_progress_bar=False, convert_to_numpy=True
        )

        print("[重建] 4/5 重建 BM25 索引...")
        self.doc_tokens = [list(text) for text in self.doc_texts]
        self.bm25 = BM25Okapi(self.doc_tokens)

        print("[重建] 5/5 重新加载查询理解别名...")
        self.query_understanding = QueryUnderstanding(
            str(Path(__file__).parent / "data" / "three_kingdoms.json")
        )

        stats = self.kg.get_stats()
        print(f"[重建] 完成：{stats['nodes']}节点 / {stats['edges']}边 / {len(self.docs)}文档")

    def _build_documents(self) -> list[dict]:
        """将知识图谱实体和关系转换为文档"""
        docs = []
        for node_id, data in self.kg.graph.nodes(data=True):
            entity_type = data.get("type", "unknown")
            text = f"{data['name']}。{data.get('description', '')}"

            neighbors = self.kg.get_neighbors(node_id)
            if neighbors:
                rel_texts = []
                for n in neighbors:
                    rel_texts.append(
                        f"{n['relation']}：{n['entity']['name']}（{n['detail']}）"
                    )
                text += " 关系：" + "；".join(rel_texts)

            docs.append({
                "id": node_id,
                "text": text,
                "metadata": {
                    "type": entity_type,
                    "name": data["name"],
                    "node_id": node_id,
                },
            })
        return docs

    # ========== 社区摘要（阶段三：GraphRAG 分层检索） ==========

    def _build_community_summaries(self) -> list[dict]:
        """
        为每个社区生成摘要文本（微软 GraphRAG 论文核心思想）
        无 LLM 时用模板摘要：社区成员 + 内部关系
        有 LLM 时调用 LLM 生成自然语言摘要
        """
        summaries = []
        node_map = self.community_info["node_map"]
        communities = self.community_info["communities"]

        for comm in communities:
            idx = comm["index"]
            members = comm["members"]

            # 收集社区内所有实体的关系
            internal_relations = []
            member_ids = {m["id"] for m in members}
            for m in members:
                neighbors = self.kg.get_neighbors(m["id"])
                for n in neighbors:
                    nid = n["entity"].get("id", "")
                    if nid in member_ids:
                        internal_relations.append(
                            f"{m['name']} --[{n['relation']}]--> {n['entity']['name']}"
                        )

            # 构建摘要文本
            member_names = [m["name"] for m in members]
            summary_text = (
                f"社区#{idx}（{comm['size']}人）：{', '.join(member_names)}。"
                f" 内部关系：{'；'.join(internal_relations[:10]) if internal_relations else '无'}"
            )

            summaries.append({
                "index": idx,
                "text": summary_text,
                "size": comm["size"],
                "members": member_names,
                "relations": internal_relations,
            })

        return summaries

    def _community_search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        社区级检索：用向量相似度找最相关的社区摘要
        这是 GraphRAG 的"分层检索"：先找社区，再找实体
        """
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        scores = np.dot(query_vec, self.community_embeddings.T)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "community_index": self.community_summaries[idx]["index"],
                "text": self.community_summaries[idx]["text"],
                "score": float(scores[idx]),
                "rank": rank,
                "members": self.community_summaries[idx]["members"],
            })
        return results

    # ========== 检索策略 ==========

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 稀疏检索（关键词精确匹配）"""
        query_tokens = list(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            hits.append({
                "id": self.doc_ids[idx],
                "text": self.doc_texts[idx],
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank,
                "metadata": self.docs[idx]["metadata"],
            })
        return hits

    def _vector_search(self, query: str, top_k: int = 10) -> list[dict]:
        """稠密向量检索（语义匹配）"""
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        similarities = np.dot(query_vec, self.doc_embeddings.T)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        hits = []
        for rank, idx in enumerate(top_indices):
            hits.append({
                "id": self.doc_ids[idx],
                "text": self.doc_texts[idx],
                "vector_score": float(similarities[idx]),
                "vector_rank": rank,
                "metadata": self.docs[idx]["metadata"],
            })
        return hits

    def _hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        混合检索：BM25 + 向量检索，使用 Reciprocal Rank Fusion (RRF) 融合分数
        RRF 公式：score = Σ 1/(k + rank_i)，k 通常取 60
        """
        bm25_hits = self._bm25_search(query, top_k=top_k * 2)
        vector_hits = self._vector_search(query, top_k=top_k * 2)

        # 构建 id → hit 映射
        all_hits = {}
        for h in bm25_hits:
            all_hits[h["id"]] = h
        for h in vector_hits:
            if h["id"] in all_hits:
                all_hits[h["id"]]["vector_score"] = h["vector_score"]
                all_hits[h["id"]]["vector_rank"] = h["vector_rank"]
            else:
                all_hits[h["id"]] = h

        # RRF 分数融合
        rrf_k = 60
        for hit_id, hit in all_hits.items():
            rrf_score = 0.0
            if "bm25_rank" in hit:
                rrf_score += 1.0 / (rrf_k + hit["bm25_rank"] + 1)
            if "vector_rank" in hit:
                rrf_score += 1.0 / (rrf_k + hit["vector_rank"] + 1)
            hit["rrf_score"] = rrf_score

        # 按 RRF 分数排序
        sorted_hits = sorted(all_hits.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_hits[:top_k]

    def _rerank(self, query: str, hits: list[dict], top_k: int = 5,
                strong_entity_names: set = None,
                neighbor_relations: dict = None,
                expanded_relations: list = None,
                passive_intent: bool = False) -> list[dict]:
        """
        重排序：Cross-Encoder 优先，回退到语义相似度
        召回阶段用双塔模型（快但粗），重排阶段用交叉编码器（慢但精）
        额外加分：
          - 查询实体加分（实体名出现在问题中）
          - 邻居关系加分（邻居与查询实体的关系匹配查询理解的关系词）
        strong_entity_names: 查询理解明确链接到的实体名，给予更大加分
        neighbor_relations: {doc_id: set((relation, direction), ...)} 邻居文档与查询实体的关系及方向
        expanded_relations: 查询理解扩展出的关系词列表
        passive_intent: 被动意图（如"X是怎么死的"），关系加分只匹配入边（查询实体是受事者）
        """
        if not hits:
            return hits

        if self.reranker_type == "cross-encoder" and self.reranker is not None:
            pairs = [(query, h["text"]) for h in hits]
            scores = self.reranker.predict(pairs)
            for i, hit in enumerate(hits):
                hit["rerank_score"] = float(scores[i])
        else:
            query_vec = self.encoder.encode([query], convert_to_numpy=True)
            doc_vecs = np.array([self.doc_embeddings[self.doc_ids.index(h["id"])] for h in hits])
            scores = np.dot(query_vec, doc_vecs.T)[0]
            for i, hit in enumerate(hits):
                hit["rerank_score"] = float(scores[i])

        # 查询实体加分
        ENTITY_BOOST = 1.0          # 普通加分：实体名出现在查询中
        STRONG_ENTITY_BOOST = 2.5   # 强加分：查询理解明确链接到的实体
        RELATION_BOOST = 2.0        # 邻居关系加分：关系匹配查询意图
        strong_entity_names = strong_entity_names or set()
        neighbor_relations = neighbor_relations or {}
        expanded_relations = expanded_relations or []

        for hit in hits:
            name = hit.get("metadata", {}).get("name", "")
            if name and name in query:
                hit["rerank_score"] += ENTITY_BOOST
                hit["entity_boost"] = True
            if name and name in strong_entity_names:
                hit["rerank_score"] += STRONG_ENTITY_BOOST
                hit["entity_boost"] = True
            # 邻居关系加分：邻居与查询实体的关系匹配查询理解的关系词
            if expanded_relations:
                rels = neighbor_relations.get(hit["id"], set())
                for rel, direction in rels:
                    # 被动意图只匹配入边（查询实体是受事者，如"关羽是怎么死的"→吕蒙擒杀关羽）
                    if passive_intent and direction != "in":
                        continue
                    if any(exp in rel or rel in exp for exp in expanded_relations):
                        hit["rerank_score"] += RELATION_BOOST
                        hit["relation_boost"] = True
                        break

        reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _multi_hop_expand(self, entity_ids: list[str], max_depth: int = 2) -> list[dict]:
        """
        阶段二：多跳图扩展（BFS），记录完整关系路径
        例如：关羽 --[敌对]--> 吕布 --[弑杀]--> 董卓
        """
        all_expanded = []
        for entity_id in entity_ids:
            hops = self.kg.multi_hop_neighbors(entity_id, max_depth=max_depth)
            all_expanded.extend(hops)

        # 去重：同一实体可能从不同源到达，保留最短路径
        seen = {}
        for item in all_expanded:
            nid = item["id"]
            if nid not in seen or item["depth"] < seen[nid]["depth"]:
                seen[nid] = item
        return list(seen.values())

    def _get_community_context(self, entity_ids: list[str]) -> list[dict]:
        """阶段二：获取命中实体所在社区的信息"""
        node_map = self.community_info["node_map"]
        communities = self.community_info["communities"]

        seen_communities = set()
        result = []
        for eid in entity_ids:
            comm_idx = node_map.get(eid)
            if comm_idx is not None and comm_idx not in seen_communities:
                seen_communities.add(comm_idx)
                comm = communities[comm_idx]
                # 只列出同社区的成员名称，不列全部
                member_names = [m["name"] for m in comm["members"][:15]]
                result.append({
                    "community_index": comm_idx,
                    "size": comm["size"],
                    "members": member_names,
                })
        return result

    def _get_subgraph_context(self, entity_ids: list[str]) -> list[dict]:
        """阶段二：提取命中实体间的内部边关系"""
        return self.kg.get_subgraph(entity_ids)

    def _try_relation_chain(self, question: str, entity_ids: list[str],
                            reranked_hits: list[dict] = None,
                            extra_relation_types: list[str] = None) -> list[dict]:
        """
        阶段二：关系推理查询
        从自然语言中提取关系链，对所有命中实体做多跳推理
        """
        # 关系关键词映射（扩展版）
        RELATION_MAP = {
            "敌对": ["敌人", "敌对", "对手", "仇敌", "死敌"],
            "君臣": ["主公", "君主", "臣子", "效忠"],
            "结义兄弟": ["兄弟", "结义"],
            "父子": ["父亲", "儿子", "之父", "之子"],
            "夫妻": ["妻子", "夫人", "妻"],
            "师徒": ["师傅", "徒弟", "师父", "学生"],
            "斩杀": ["杀了", "斩杀", "斩", "杀", "擒杀"],
            "归降": ["投降", "归降", "归顺"],
            "弑杀": ["弑杀", "杀害", "谋杀", "弑", "杀了", "杀"],
        }

        # 检测问题中包含的关系链
        detected_relations = []
        for rel_type, keywords in RELATION_MAP.items():
            for kw in keywords:
                if kw in question:
                    detected_relations.append(rel_type)
                    break

        if not detected_relations or not entity_ids:
            # 如果查询扩展提供了额外关系词，使用它们
            if extra_relation_types and entity_ids:
                detected_relations = extra_relation_types
            else:
                return []

        # 合并查询扩展的关系词
        if extra_relation_types:
            for rel in extra_relation_types:
                if rel not in detected_relations:
                    detected_relations.append(rel)

        # 优先用问题中提到的实体名匹配
        candidate_ids = list(entity_ids)
        name_matches = []
        other_ids = []
        for eid in candidate_ids:
            entity = self.kg.get_entity(eid)
            name = entity.get("name", "") if entity else ""
            if name and name in question:
                name_matches.append(eid)
            else:
                other_ids.append(eid)
        candidate_ids = name_matches + other_ids

        # 对候选实体尝试关系链查询
        # 如果有名字匹配的实体，只查那些；否则查所有
        search_ids = candidate_ids[:5] if not name_matches else name_matches[:3]
        best_result = []
        for eid in search_ids:
            # 单跳查询：每个关系类型独立查
            for rel in detected_relations:
                chain = self.kg.relation_chain_query(eid, [rel])
                if len(chain) > len(best_result):
                    best_result = chain
            # 多跳链查询：关系链
            if len(detected_relations) >= 2:
                chain = self.kg.relation_chain_query(eid, detected_relations[:2])
                if len(chain) > len(best_result):
                    best_result = chain

        return best_result

    # ========== 主查询接口 ==========

    def query(
        self,
        question: str,
        top_k: int = 5,
        expand_graph: bool = True,
    ) -> dict:
        """
        完整 RAG 检索流程 v4（阶段三：查询理解 + 三路融合）：
        0. 查询理解：实体链接 + 查询扩展 + 意图分类
        1. BM25 + 向量 混合召回（实体级）
        2. Cross-Encoder / 语义重排序
        3. 社区级检索（GraphRAG 分层检索，社区摘要匹配）
        4. 多跳图扩展（深度2，带路径）
        5. 子图关系提取
        6. 关系推理（使用扩展后的关系词）
        三路融合：实体检索 + 图检索 + 社区检索 统一打分
        """
        # 0. 查询理解（实体链接 + 查询扩展 + 意图分类）
        qu_result = self.query_understanding.understand(question)
        search_query = qu_result["rewritten_query"]  # 用改写后的查询做检索
        expanded_relations = qu_result["expanded_relations"]

        # 1. 混合召回（实体级，用改写后的查询）
        recall_hits = self._hybrid_search(search_query, top_k=top_k * 2)

        # 1.5 实体注入：对于对比类查询（X和Y），确保查询理解链接到的实体都被召回
        linked_names = {r["standard_name"] for r in qu_result.get("entity_replacements", [])}
        # 也检查原始查询中直接出现的实体名
        for doc in self.docs:
            name = doc.get("metadata", {}).get("name", "")
            if name and name in search_query:
                linked_names.add(name)
        existing_ids = {h["id"] for h in recall_hits}
        linked_entity_ids = []  # 收集 linked 实体 ID，供 1.6 邻居注入复用
        for name in linked_names:
            entities = self.kg.search_by_name(name)
            for e in entities:
                linked_entity_ids.append(e["id"])
                if e["id"] not in existing_ids:
                    # 找到对应的文档
                    for doc in self.docs:
                        if doc["id"] == e["id"]:
                            recall_hits.append(doc)
                            existing_ids.add(e["id"])
                            break

        # 1.6 邻居注入：对查询中明确提到的实体，注入其 1 跳邻居到召回候选集
        # 解决"答案实体未召回"问题：单跳关系问题（如"关羽是怎么死的"→吕蒙、"定军山斩夏侯渊"→黄忠）
        # 答案实体的描述里可能没有查询关键词，BM25+向量召回拿不到；但它们是主语实体的直接邻居
        # 仅注入 1 跳邻居，避免候选集膨胀；邻居是否相关由后续 rerank 关系加分判断
        # neighbor_relations 记录 {doc_id: set((relation, direction), ...)}，direction: out=查询实体施事, in=查询实体受事
        neighbor_relations = {}
        for eid in linked_entity_ids:
            for nb in self.kg.get_neighbors(eid, depth=1):
                nb_id = nb["entity"]["id"]
                # 记录关系和方向（无论邻居是否已存在，用于后续关系加分）
                neighbor_relations.setdefault(nb_id, set()).add((nb["relation"], nb["direction"]))
                if nb_id not in existing_ids:
                    for doc in self.docs:
                        if doc["id"] == nb_id:
                            recall_hits.append(doc)
                            existing_ids.add(nb_id)
                            break

        # 检测被动意图：查询问"X是怎么死的/谁杀了X"时，X是受事者，关系加分只匹配入边（别人作用于X）
        # 避免把"关羽斩文丑"（出边）和"吕蒙擒杀关羽"（入边）混淆
        passive_intent = any(kw in search_query for kw in [
            "怎么死", "如何死", "是怎么死", "被害", "被杀", "被斩", "被擒",
            "身亡", "阵亡", "谁杀", "谁斩", "谁擒", "谁害",
        ])

        # 2. 重排序（用改写后的查询，强匹配实体给予更大加分，邻居关系匹配给予关系加分）
        reranked_hits = self._rerank(
            search_query, recall_hits, top_k=top_k,
            strong_entity_names=linked_names,
            neighbor_relations=neighbor_relations,
            expanded_relations=expanded_relations,
            passive_intent=passive_intent,
        )

        entity_ids = [h["id"] for h in reranked_hits]

        # 3. 社区级检索（阶段三：分层检索）
        community_search_results = self._community_search(search_query, top_k=3)

        # 4. 多跳图扩展
        graph_context = []
        if expand_graph and reranked_hits:
            graph_context = self._multi_hop_expand(entity_ids, max_depth=self.max_hop_depth)

        # 5. 社区上下文（基于命中实体所在社区）
        community_context = self._get_community_context(entity_ids)

        # 6. 子图关系
        subgraph_context = self._get_subgraph_context(entity_ids)

        # 7. 关系推理（使用查询扩展后的关系词 + 图扩展中发现的实体）
        chain_candidate_ids = list(entity_ids)
        if graph_context:
            for g in graph_context:
                if g.get("depth", 99) <= 1:
                    chain_candidate_ids.append(g["id"])
        chain_context = self._try_relation_chain(
            search_query, chain_candidate_ids, reranked_hits,
            extra_relation_types=expanded_relations,
        )

        # 构建上下文
        context_parts = []
        context_parts.append("=== 重排序后的匹配实体 ===")
        for i, h in enumerate(reranked_hits):
            context_parts.append(
                f"[{i+1}] {h['metadata']['name']} "
                f"(RRF: {h.get('rrf_score', 0):.4f}, "
                f"Rerank: {h.get('rerank_score', 0):.4f})"
            )
            context_parts.append(f"    {h['text']}")

        if graph_context:
            context_parts.append("\n=== 多跳图扩展（深度2） ===")
            for g in graph_context:
                entity = g["entity"]
                path_str = " ".join(g.get("path", []))
                context_parts.append(
                    f"- [跳数{g['depth']}] {path_str}"
                )

        if subgraph_context:
            context_parts.append("\n=== 命中实体间的直接关系 ===")
            for e in subgraph_context:
                context_parts.append(
                    f"- {e['source_name']} --[{e['relation']}]--> {e['target_name']}"
                )

        if community_context:
            context_parts.append("\n=== 社区上下文 ===")
            for c in community_context:
                context_parts.append(
                    f"- 社区#{c['community_index']}（{c['size']}人）：{', '.join(c['members'])}"
                )

        if chain_context:
            context_parts.append("\n=== 关系推理链 ===")
            for c in chain_context:
                context_parts.append(
                    f"- [跳{c['hop']}] {c['source_name']} --[{c['relation']}]--> {c['entity_name']}：{c['detail']}"
                )

        if community_search_results:
            context_parts.append("\n=== 社区级检索（分层检索 Top-3） ===")
            for cr in community_search_results:
                context_parts.append(
                    f"- 社区#{cr['community_index']} (Score={cr['score']:.4f})：{cr['text'][:100]}"
                )

        context = "\n".join(context_parts)

        return {
            "question": question,
            "query_understanding": qu_result,
            "recall_hits": recall_hits,
            "reranked_hits": reranked_hits,
            "graph_context": graph_context,
            "community_context": community_context,
            "community_search_results": community_search_results,
            "subgraph_context": subgraph_context,
            "chain_context": chain_context,
            "context": context,
        }

    # ========== LLM 回答生成 ==========

    def _init_llm(self):
        """初始化 LLM，支持 DeepSeek 和 OpenAI 兼容 API；无 key 则返回 None"""
        # 优先支持 DeepSeek：只需设置 DEEPSEEK_API_KEY 即可自动配置
        deepseek_key = self.llm_config.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key and not self.llm_config.get("base_url"):
            api_key = deepseek_key
            base_url = "https://api.deepseek.com"
            model = self.llm_config.get("model", "deepseek-chat")
        else:
            # 通用 OpenAI 兼容接口
            api_key = self.llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            base_url = self.llm_config.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
            model = self.llm_config.get("model", "gpt-3.5-turbo")

        if not api_key:
            print("  未检测到 LLM API key，将使用增强模板模式")
            print("  启用 LLM 方式之一：")
            print("    1. 设置 DEEPSEEK_API_KEY 环境变量（推荐，自动配置 DeepSeek）")
            print("    2. 设置 OPENAI_API_KEY + OPENAI_BASE_URL 环境变量")
            return None

        try:
            from openai import OpenAI
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            print(f"  LLM 就绪: {model} @ {base_url or 'OpenAI默认'}")
            return {"client": client, "model": model}
        except ImportError:
            print("  openai 库未安装，使用模板模式（pip install openai 启用 LLM）")
            return None
        except Exception as e:
            print(f"  LLM 初始化失败: {e}，使用模板模式")
            return None

    def _llm_generate(self, question: str, context: str) -> str:
        """调用 LLM 基于检索上下文生成自然语言回答"""
        system_prompt = """你是一个三国演义知识图谱专家。你的任务是基于知识图谱的检索结果回答用户问题。

严格规则：
1. 只能基于【检索上下文】中的信息回答，绝对不要编造或引入外部知识
2. 如果检索上下文中没有与问题相关的信息，直接回答："知识图谱中没有找到相关记录。"
3. 善用"关系推理链"和"命中实体间的直接关系"进行多跳推理
   - 例如问题"X的敌人的主公是谁"，先从推理链/子图关系找到X的敌人，再找敌人的主公
4. 回答简洁流畅，突出关键人物关系和事件
5. 涉及多跳推理时，简要说明推理路径"""

        user_prompt = f"""【检索上下文】
{context}

【用户问题】
{question}

请基于上述检索上下文回答："""

        try:
            resp = self.llm["client"].chat.completions.create(
                model=self.llm["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM 调用失败: {e}]\n\n" + self._template_answer(question)

    def _template_answer(self, question: str) -> str:
        """无 LLM 时的增强模板回答（阶段三：推理链优先 + 三路融合）"""
        result = self.query(question)
        hits = result["reranked_hits"]

        if not hits:
            return "未找到与您问题相关的知识图谱信息。"

        lines = []
        chain = result.get("chain_context", [])
        qu = result.get("query_understanding", {})
        expanded_relations = qu.get("expanded_relations", [])

        # 检测是否有明确的关系意图（查询扩展到了关系词）
        has_relation_intent = len(expanded_relations) > 0

        # 如果有明确关系意图，但推理链和子图都找不到相关关系
        # 说明知识图谱中没有这个数据，直接回答"不知道"
        if has_relation_intent:
            # 检查推理链中是否有以排名第一实体为起点的关系
            primary_name = hits[0]["metadata"]["name"]
            chain_for_primary = [c for c in chain if c.get("source_name") == primary_name]

            # 也检查子图中是否有以排名第一实体为起点的关系
            subgraph = result.get("subgraph_context", [])
            subgraph_for_primary = [
                e for e in subgraph
                if e["source_name"] == primary_name or e["target_name"] == primary_name
            ]

            if not chain_for_primary and not subgraph_for_primary:
                return (
                    f"知识图谱中没有找到 {primary_name} 的相关关系记录。\n"
                    f"（检测到关系意图：{', '.join(expanded_relations)}，"
                    f"但图谱中无对应数据）\n\n"
                    f"【{primary_name}】{hits[0]['text'].split('关系：')[0].strip()}"
                )

        # 如果有 2 跳推理链结果，说明是多跳推理问题，用推理链作为回答主体
        hop2_results = [c for c in chain if c.get("hop") == 2]
        if hop2_results:
            lines.append(f"根据知识图谱推理，回答「{question}」：\n")

            # 按 1 跳实体分组展示推理链
            hop1_map = {}  # source_name → list of hop1 results
            for c in chain:
                if c.get("hop") == 1:
                    key = c["source_name"]
                    if key not in hop1_map:
                        hop1_map[key] = []
                    hop1_map[key].append(c)

            for source_name, hop1_list in hop1_map.items():
                lines.append(f"■ {source_name} 的相关关系：")
                for h1 in hop1_list:
                    lines.append(
                        f"  → [{h1['relation']}] {h1['entity_name']}：{h1['detail']}"
                    )
                    # 找对应的 2 跳结果
                    for h2 in hop2_results:
                        if h2.get("source_id") == h1["entity_id"]:
                            lines.append(
                                f"    →→ [{h2['relation']}] {h2['entity_name']}：{h2['detail']}"
                            )
                lines.append("")

            # 总结答案
            answer_names = [h["entity_name"] for h in hop2_results]
            lines.append(f"∴ 答案：{', '.join(answer_names)}")

        else:
            # 没有多跳推理链
            subgraph = result.get("subgraph_context", [])

            # 从查询理解中获取用户明确提到的实体名
            qu = result.get("query_understanding", {})
            query_entity_names = set()
            # 实体链接替换后的标准名
            for r in qu.get("entity_replacements", []):
                query_entity_names.add(r["standard_name"])
            # 也检查改写后查询中直接出现的实体名
            rewritten = qu.get("rewritten_query", question)
            for h in hits:
                name = h["metadata"]["name"]
                if name in rewritten:
                    query_entity_names.add(name)

            # 只筛选查询中明确提到的实体之间的关系
            core_relations = [
                e for e in subgraph
                if e["source_name"] in query_entity_names
                and e["target_name"] in query_entity_names
            ]

            if core_relations and len(query_entity_names) >= 2:
                # 对比/关系类查询：聚焦展示命中实体间的直接关系
                lines.append(f"根据知识图谱，回答「{question}」：\n")

                for e in core_relations:
                    lines.append(f"■ {e['source_name']} --[{e['relation']}]--> {e['target_name']}")
                    # 找 detail
                    for h in hits:
                        if h["metadata"]["name"] == e["source_name"]:
                            # 从文档文本中提取该关系的详情
                            text = h.get("text", "")
                            if e["target_name"] in text:
                                # 截取包含目标实体的片段
                                idx = text.find(e["target_name"])
                                start = max(0, text.rfind("；", 0, idx) + 1)
                                end = text.find("；", idx)
                                if end == -1:
                                    end = min(len(text), idx + 60)
                                detail = text[start:end].strip()
                                lines.append(f"  详情：{detail}")
                            break

                # 补充：列出查询中提到的实体简介
                lines.append("")
                for h in hits:
                    name = h["metadata"]["name"]
                    if name in query_entity_names:
                        desc = h["text"].split("关系：")[0].strip()
                        lines.append(f"【{name}】{desc}")

            else:
                # 普通查询：以排名第一的实体为主体
                primary = hits[0]
                lines.append(f"【{primary['metadata']['name']}】")
                lines.append(primary["text"])

                # 多跳图扩展（带路径）
                if result["graph_context"]:
                    lines.append(f"\n多跳关联（路径）：")
                    for g in result["graph_context"][:10]:
                        path_str = " ".join(g.get("path", []))
                        lines.append(f"  • [跳{g['depth']}] {path_str}")

                # 命中实体间的直接关系
                if subgraph:
                    lines.append(f"\n命中实体间关系：")
                    for e in subgraph:
                        lines.append(f"  • {e['source_name']} --[{e['relation']}]--> {e['target_name']}")

                # 关系推理链（单跳）
                if chain:
                    lines.append(f"\n关系推理链：")
                    for c in chain:
                        lines.append(f"  • [跳{c['hop']}] {c['source_name']} --[{c['relation']}]--> {c['entity_name']}：{c['detail']}")

        # 社区信息
        if result.get("community_context"):
            lines.append(f"\n所属社区：")
            for c in result["community_context"]:
                lines.append(f"  • 社区#{c['community_index']}（{c['size']}人）：{', '.join(c['members'][:8])}")

        return "\n".join(lines)

    def answer(self, question: str, top_k: int = 5) -> str:
        """生成回答：有 LLM 用 LLM，否则用模板"""
        if self.llm:
            result = self.query(question, top_k=top_k)
            return self._llm_generate(question, result["context"])
        else:
            return self._template_answer(question)