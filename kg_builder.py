"""
知识图谱构建器 - 基于 NetworkX 构建三国演义知识图谱
"""
import json
import webbrowser
import networkx as nx
from pathlib import Path
from typing import Optional


class KnowledgeGraph:
    """三国演义知识图谱"""

    def __init__(self, data_path: str = "data/three_kingdoms.json"):
        self.graph = nx.DiGraph()
        self.data_path = Path(data_path)
        self._load_data()
        self._build_graph()

    def _load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def _build_graph(self):
        # 添加实体节点
        for char in self.data["entities"]["characters"]:
            self.graph.add_node(
                char["id"],
                type="character",
                name=char["name"],
                faction=char["faction"],
                description=char["description"],
            )

        for faction in self.data["entities"]["factions"]:
            self.graph.add_node(
                faction["id"],
                type="faction",
                name=faction["name"],
                description=faction["description"],
            )

        for event in self.data["entities"]["events"]:
            self.graph.add_node(
                event["id"],
                type="event",
                name=event["name"],
                description=event["description"],
            )

        # 添加关系边
        for rel in self.data["relationships"]:
            self.graph.add_edge(
                rel["source"],
                rel["target"],
                relation=rel["relation"],
                detail=rel["detail"],
            )

    def get_entity(self, entity_id: str) -> Optional[dict]:
        """获取单个实体信息"""
        if entity_id in self.graph.nodes:
            return dict(self.graph.nodes[entity_id])
        return None

    def get_neighbors(self, entity_id: str, depth: int = 1) -> list[dict]:
        """获取实体的邻居（关联实体）"""
        if entity_id not in self.graph.nodes:
            return []

        neighbors = []
        # 出边
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            node_data = dict(self.graph.nodes[target])
            node_data["id"] = target
            neighbors.append({
                "entity": node_data,
                "relation": data["relation"],
                "detail": data["detail"],
                "direction": "out",
            })
        # 入边
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            node_data = dict(self.graph.nodes[source])
            node_data["id"] = source
            neighbors.append({
                "entity": node_data,
                "relation": data["relation"],
                "detail": data["detail"],
                "direction": "in",
            })
        return neighbors

    def find_path(self, source_id: str, target_id: str) -> list[dict]:
        """查找两个实体之间的最短路径"""
        try:
            path = nx.shortest_path(self.graph, source=source_id, target=target_id)
            result = []
            for i in range(len(path) - 1):
                edge_data = self.graph.edges[path[i], path[i + 1]]
                result.append({
                    "from": dict(self.graph.nodes[path[i]]),
                    "to": dict(self.graph.nodes[path[i + 1]]),
                    "relation": edge_data["relation"],
                    "detail": edge_data["detail"],
                })
            return result
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_subgraph(self, entity_ids: list[str]) -> "KnowledgeGraph":
        """获取子图"""
        subgraph = self.graph.subgraph(entity_ids)
        kg = KnowledgeGraph.__new__(KnowledgeGraph)
        kg.graph = subgraph
        kg.data = self.data
        kg.data_path = self.data_path
        return kg

    def search_by_name(self, name: str) -> list[dict]:
        """按名称搜索实体"""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            if name.lower() in data["name"].lower():
                results.append({"id": node_id, **data})
        return results

    def get_faction_members(self, faction_name: str) -> list[dict]:
        """获取势力下的所有人物"""
        members = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "character" and data.get("faction") == faction_name:
                members.append({"id": node_id, **data})
        return members

    def get_all_entities(self, entity_type: str = None) -> list[dict]:
        """获取所有实体"""
        entities = []
        for node_id, data in self.graph.nodes(data=True):
            if entity_type is None or data.get("type") == entity_type:
                entities.append({"id": node_id, **data})
        return entities

    def get_stats(self) -> dict:
        """获取图谱统计信息"""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "characters": sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "character"),
            "factions": sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "faction"),
            "events": sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "event"),
        }

    # ========== 增量更新 ==========

    def add_entity(self, entity_type: str, entity_id: str, name: str,
                   description: str, faction: str = None) -> bool:
        """
        添加实体节点
        entity_type: character / faction / event
        返回：是否添加成功（已存在则返回 False）
        """
        if entity_id in self.graph.nodes:
            return False
        attrs = {"type": entity_type, "name": name, "description": description}
        if faction:
            attrs["faction"] = faction
        self.graph.add_node(entity_id, **attrs)
        return True

    def add_relation(self, source: str, target: str,
                     relation: str, detail: str) -> bool:
        """
        添加关系边
        如果边已存在（相同 source→target→relation），返回 False
        """
        if source not in self.graph.nodes or target not in self.graph.nodes:
            return False
        # 检查是否已存在相同的边
        for _, _, data in self.graph.edges(source, data=True):
            if data.get("relation") == relation:
                return False
        self.graph.add_edge(source, target, relation=relation, detail=detail)
        return True

    def remove_entity(self, entity_id: str) -> bool:
        """删除实体（同时删除所有关联边）"""
        if entity_id not in self.graph.nodes:
            return False
        self.graph.remove_node(entity_id)  # NetworkX 自动删除关联边
        return True

    def remove_relation(self, source: str, target: str,
                        relation: str = None) -> int:
        """
        删除关系边
        relation=None 时删除 source→target 的所有边
        返回：删除的边数
        """
        if relation is None:
            if self.graph.has_edge(source, target):
                n = self.graph.number_of_edges(source, target)
                self.graph.remove_edges_from([(source, target)])
                return n
            return 0
        # 删除特定 relation 的边
        edges_to_remove = []
        for s, t, data in self.graph.edges(source, data=True):
            if t == target and data.get("relation") == relation:
                edges_to_remove.append((s, t))
        for e in edges_to_remove:
            self.graph.remove_edge(*e)
        return len(edges_to_remove)

    def save_to_file(self, path: str = None) -> str:
        """
        将当前图谱状态保存回 JSON 文件
        返回保存的文件路径
        """
        save_path = path or str(self.data_path)
        data = {"entities": {"characters": [], "factions": [], "events": []},
                "relationships": []}

        for node_id, attrs in self.graph.nodes(data=True):
            etype = attrs.get("type", "character")
            entry = {"id": node_id, "name": attrs.get("name", ""),
                     "type": etype, "description": attrs.get("description", "")}
            if etype == "character" and "faction" in attrs:
                entry["faction"] = attrs["faction"]
            key = {"character": "characters", "faction": "factions",
                   "event": "events"}.get(etype, "characters")
            data["entities"][key].append(entry)

        for source, target, attrs in self.graph.edges(data=True):
            data["relationships"].append({
                "source": source, "target": target,
                "relation": attrs.get("relation", ""),
                "detail": attrs.get("detail", ""),
            })

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return save_path

    # ========== 阶段二：图算法 ==========

    def detect_communities(self) -> dict:
        """
        社区发现：用 Louvain 贪心模块度算法将图谱分群
        返回 {node_id: community_index} 映射和社区列表
        """
        # 将有向图转为无向图做社区发现
        undirected = self.graph.to_undirected()
        communities = nx.community.greedy_modularity_communities(undirected)

        node_community = {}
        community_list = []
        for i, comm in enumerate(communities):
            members = []
            for node_id in comm:
                data = self.graph.nodes[node_id]
                node_community[node_id] = i
                members.append({
                    "id": node_id,
                    "name": data.get("name", ""),
                    "type": data.get("type", ""),
                    "faction": data.get("faction", ""),
                })
            community_list.append({
                "index": i,
                "size": len(members),
                "members": members,
            })

        return {"node_map": node_community, "communities": community_list}

    def get_subgraph(self, entity_ids: list[str]) -> list[dict]:
        """
        提取给定实体集合的内部子图（实体之间的边）
        用于发现检索结果中实体间的直接关系
        """
        entity_set = set(entity_ids)
        edges = []
        for u, v, d in self.graph.edges(data=True):
            if u in entity_set and v in entity_set:
                edges.append({
                    "source": u,
                    "source_name": self.graph.nodes[u].get("name", u),
                    "target": v,
                    "target_name": self.graph.nodes[v].get("name", v),
                    "relation": d["relation"],
                    "detail": d["detail"],
                })
        return edges

    def multi_hop_neighbors(self, entity_id: str, max_depth: int = 2) -> list[dict]:
        """
        多跳 BFS 遍历：从 entity_id 出发，记录每跳的关系路径
        返回 [{id, entity, relation, detail, depth, path: [节点名链]}]
        """
        visited = {entity_id}
        result = []
        frontier = [entity_id]
        # 记录路径：node_id → [节点名列表]
        paths = {entity_id: [self.graph.nodes[entity_id].get("name", entity_id)]}

        for depth in range(1, max_depth + 1):
            next_frontier = []
            for current in frontier:
                neighbors = self.get_neighbors(current)
                for n in neighbors:
                    nid = n["entity"].get("id", "")
                    if not nid or nid in visited:
                        continue
                    visited.add(nid)
                    next_frontier.append(nid)
                    # 构建路径
                    current_path = paths[current].copy()
                    current_path.append(
                        f"--[{n['relation']}]-->{n['entity']['name']}"
                    )
                    paths[nid] = current_path

                    result.append({
                        "id": nid,
                        "entity": n["entity"],
                        "relation": n["relation"],
                        "detail": n["detail"],
                        "source_node": current,
                        "depth": depth,
                        "path": current_path,
                    })
            frontier = next_frontier
            if not frontier:
                break

        return result

    def relation_chain_query(self, entity_id: str, relation_types: list[str]) -> list[dict]:
        """
        关系推理：按指定关系链遍历图谱
        每一跳尝试所有匹配的关系类型（灵活匹配）
        例如：relation_chain_query("guanyu", ["敌对"]) → 关羽的敌人
              relation_chain_query("guanyu", ["敌对", "君臣"]) → 关羽的敌人的主公
              relation_chain_query("zhugeliang", ["敌对"]) → 诸葛亮的死敌（"死敌"含"敌"字也匹配）
        """
        current_entities = [entity_id]
        chain_result = []
        visited_per_hop = {0: {entity_id}}

        for hop, rel_type in enumerate(relation_types):
            next_entities = []
            visited_per_hop[hop + 1] = set()
            for eid in current_entities:
                neighbors = self.get_neighbors(eid)
                for n in neighbors:
                    # 灵活匹配：关系类型的关键字符出现在关系字符串中即可
                    relation_str = n["relation"]
                    matched = (
                        rel_type in relation_str
                        or relation_str in rel_type
                        or rel_type[0] in relation_str  # 首字匹配：敌对→死敌, 弑杀→义父子→弑杀
                    )
                    if matched:
                        nid = n["entity"].get("id", "")
                        if nid and nid not in visited_per_hop[hop + 1]:
                            visited_per_hop[hop + 1].add(nid)
                            next_entities.append(nid)
                            chain_result.append({
                                "hop": hop + 1,
                                "relation": n["relation"],
                                "entity_id": nid,
                                "entity_name": n["entity"]["name"],
                                "entity": n["entity"],
                                "detail": n["detail"],
                                "source_id": eid,
                                "source_name": self.graph.nodes[eid].get("name", eid),
                            })
            current_entities = list(set(next_entities))
            if not current_entities:
                break

        return chain_result

    # ========== 可视化 ==========

    # 势力配色
    FACTION_COLORS = {
        "蜀汉": "#FF6B6B",
        "曹魏": "#4ECDC4",
        "东吴": "#45B7D1",
        "群雄": "#96CEB4",
    }
    TYPE_SHAPES = {
        "character": "o",
        "faction": "s",
        "event": "D",
    }
    TYPE_SIZES = {
        "character": 800,
        "faction": 1400,
        "event": 1000,
    }

    def visualize_matplotlib(self, output_path: str = "data/kg_visual.png"):
        """使用 matplotlib 生成静态知识图谱图片"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        ax.set_title("三国演义 · 知识图谱", fontsize=20, fontweight="bold", pad=20)

        # 使用 spring_layout 布局
        pos = nx.spring_layout(self.graph, k=2.5, iterations=50, seed=42)

        # 按类型分别绘制
        for node_type in ["character", "faction", "event"]:
            nodes = [n for n, d in self.graph.nodes(data=True) if d.get("type") == node_type]
            if not nodes:
                continue

            if node_type == "character":
                colors = [
                    self.FACTION_COLORS.get(
                        self.graph.nodes[n].get("faction", "群雄"), "#CCCCCC"
                    )
                    for n in nodes
                ]
            elif node_type == "faction":
                colors = [
                    self.FACTION_COLORS.get(self.graph.nodes[n]["name"], "#CCCCCC")
                    for n in nodes
                ]
            else:
                colors = "#FFD93D"

            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=nodes,
                node_color=colors,
                node_size=self.TYPE_SIZES.get(node_type, 800),
                node_shape=self.TYPE_SHAPES.get(node_type, "o"),
                alpha=0.9,
                edgecolors="white",
                linewidths=1.5,
                ax=ax,
            )

        # 绘制边
        edge_labels = {}
        for u, v, d in self.graph.edges(data=True):
            edge_labels[(u, v)] = d["relation"]

        nx.draw_networkx_edges(
            self.graph, pos,
            alpha=0.3,
            edge_color="#888888",
            arrows=True,
            arrowsize=12,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

        # 节点标签（加白色描边确保在彩色节点上可读）
        labels = {n: d["name"] for n, d in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(
            self.graph, pos, labels,
            font_size=10,
            font_weight="bold",
            font_color="#1a1a2e",
            font_family="Microsoft YaHei",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#333", alpha=0.85, linewidth=0.5),
            ax=ax,
        )

        # 边标签（加白色背景框确保可读性）
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color="#333333",
            alpha=0.9,
            label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
            ax=ax,
        )

        # 图例
        legend_patches = [
            mpatches.Patch(color=color, label=name)
            for name, color in self.FACTION_COLORS.items()
        ]
        legend_patches.append(mpatches.Patch(color="#FFD93D", label="事件"))
        ax.legend(handles=legend_patches, loc="upper right", fontsize=10)

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"静态图谱已保存到: {output_path}")
        return output_path

    def visualize_html(self, output_path: str = "data/kg_interactive.html", open_browser: bool = True):
        """生成交互式 HTML 知识图谱（纯 HTML + vis.js，无需 pyvis）"""
        import json as _json

        # 构建节点数据
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("type", "character")
            faction = data.get("faction", "")
            color = self.FACTION_COLORS.get(faction, "#FFD93D")

            if node_type == "faction":
                color = self.FACTION_COLORS.get(data["name"], "#CCCCCC")
                shape = "star"
            elif node_type == "event":
                color = "#FFD93D"
                shape = "diamond"
            else:
                shape = "dot"

            # 节点字体：白色文字 + 黑色描边，确保在任何底色上都清晰
            nodes.append({
                "id": node_id,
                "label": data["name"],
                "title": f"<b>{data['name']}</b><br>{data.get('description', '')[:120]}",
                "color": {"background": color, "border": "#ffffff", "highlight": {"background": color, "border": "#e94560"}},
                "shape": shape,
                "type": node_type,
                "font": {"color": "#ffffff", "size": 16, "face": "Microsoft YaHei", "strokeWidth": 3, "strokeColor": "#000000"},
            })

        # 构建边数据
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "label": d["relation"],
                "title": d["detail"],
                "arrows": "to",
            })

        graph_data = _json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)

        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>三国演义 · 知识图谱</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Microsoft YaHei", "PingFang SC", sans-serif; background: #1a1a2e; overflow: hidden; }}
  #header {{
    position: absolute; top: 0; left: 0; right: 0; z-index: 10;
    padding: 12px 24px;
    background: linear-gradient(135deg, #16213e, #0f3460);
    color: #e94560; font-size: 20px; font-weight: bold;
    display: flex; align-items: center; gap: 12px;
    box-shadow: 0 2px 20px rgba(0,0,0,0.5);
  }}
  #header .subtitle {{ font-size: 13px; color: #a0a0b0; font-weight: normal; }}
  #legend {{
    position: absolute; bottom: 24px; left: 24px; z-index: 10;
    background: rgba(22,33,62,0.9); border-radius: 10px;
    padding: 14px 18px; color: #ddd; font-size: 13px;
    backdrop-filter: blur(10px);
  }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .legend-diamond {{ width: 12px; height: 12px; transform: rotate(45deg); }}
  .legend-star {{ width: 14px; height: 14px; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%); }}
  #info {{ position: absolute; top: 80px; right: 20px; z-index: 10; display: none; }}
  #mynetwork {{ width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<div id="header">
  <span>🏯</span> 三国演义 · 知识图谱
  <span class="subtitle">| 拖拽节点 | 滚轮缩放 | 悬停查看详情</span>
</div>
<div id="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#FF6B6B"></div> 蜀汉</div>
  <div class="legend-item"><div class="legend-dot" style="background:#4ECDC4"></div> 曹魏</div>
  <div class="legend-item"><div class="legend-dot" style="background:#45B7D1"></div> 东吴</div>
  <div class="legend-item"><div class="legend-dot" style="background:#96CEB4"></div> 群雄</div>
  <div class="legend-item"><div class="legend-diamond" style="background:#FFD93D"></div> 事件</div>
  <div class="legend-item"><div class="legend-star" style="background:#aaa"></div> 势力</div>
</div>
<div id="mynetwork"></div>
<script>
var data = {graph_data};
var nodes = new vis.DataSet(data.nodes);
var edges = new vis.DataSet(data.edges);
var container = document.getElementById("mynetwork");
var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, {{
  nodes: {{ font: {{ size: 16, face: "Microsoft YaHei", color: "#ffffff", strokeWidth: 3, strokeColor: "#000000" }}, borderWidth: 2, borderWidthSelected: 4 }},
  edges: {{ color: {{ color: "#777", highlight: "#e94560", opacity: 0.6 }}, smooth: {{ type: "curvedCW", roundness: 0.2 }}, font: {{ size: 11, color: "#ffffff", face: "Microsoft YaHei", strokeWidth: 3, strokeColor: "#000000", align: "middle" }}, arrows: {{ to: {{ scaleFactor: 0.8 }} }} }},
  physics: {{ barnesHut: {{ gravitationalConstant: -3000, centralGravity: 0.3, springLength: 200, springConstant: 0.04, damping: 0.3 }}, minVelocity: 0.75, stabilization: {{ iterations: 200 }} }},
  interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true, keyboard: true }}
}});
</script>
</body>
</html>'''

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"交互式图谱已保存到: {output_path}")

        if open_browser:
            webbrowser.open(f"file:///{Path(output_path).resolve().as_posix()}")
            print("已在浏览器中打开")

        return output_path