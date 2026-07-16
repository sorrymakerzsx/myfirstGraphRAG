"""
阶段三：Neo4j 图数据库适配器
保持与 KnowledgeGraph (NetworkX) 相同的接口，实现无缝切换

使用方式：
1. 安装 Neo4j Desktop 或 Docker 运行 Neo4j
2. pip install neo4j
3. 修改 main.py 中 KnowledgeGraph → Neo4jKnowledgeGraph

接口完全兼容：get_entity / get_neighbors / search_by_name / detect_communities 等
"""

from typing import Optional
import json
from pathlib import Path


class Neo4jKnowledgeGraph:
    """Neo4j 版知识图谱，接口与 NetworkX 版 KnowledgeGraph 完全兼容"""

    def __init__(self, data_path: str = None, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "password"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.data_path = data_path
        self.data = {}

        # 如果提供了数据文件路径，自动导入数据
        if data_path:
            self._import_from_json(data_path)

    def close(self):
        self.driver.close()

    def _import_from_json(self, data_path: str):
        """从 JSON 文件导入数据到 Neo4j"""
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        with self.driver.session() as session:
            # 清空已有数据
            session.run("MATCH (n) DETACH DELETE n")

            # 创建实体节点
            for char in self.data.get("entities", {}).get("characters", []):
                session.run(
                    "CREATE (n:Character {id: $id, name: $name, "
                    "description: $desc, faction: $faction, type: 'character'})",
                    id=char["id"], name=char["name"],
                    desc=char.get("description", ""),
                    faction=char.get("faction", ""),
                )

            for faction in self.data.get("entities", {}).get("factions", []):
                session.run(
                    "CREATE (n:Faction {id: $id, name: $name, "
                    "description: $desc, type: 'faction'})",
                    id=faction["id"], name=faction["name"],
                    desc=faction.get("description", ""),
                )

            for event in self.data.get("entities", {}).get("events", []):
                session.run(
                    "CREATE (n:Event {id: $id, name: $name, "
                    "description: $desc, type: 'event'})",
                    id=event["id"], name=event["name"],
                    desc=event.get("description", ""),
                )

            # 创建关系边
            for rel in self.data.get("relationships", []):
                session.run(
                    "MATCH (a {id: $source}), (b {id: $target}) "
                    "CREATE (a)-[r:RELATION {relation: $rel, detail: $detail}]->(b)",
                    source=rel["source"], target=rel["target"],
                    rel=rel["relation"], detail=rel.get("detail", ""),
                )

    # ========== 查询接口（与 NetworkX 版完全一致） ==========

    def get_entity(self, entity_id: str) -> Optional[dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n {id: $eid}) RETURN n", eid=entity_id
            )
            record = result.single()
            if record:
                data = dict(record["n"])
                data["id"] = entity_id
                return data
            return None

    def get_neighbors(self, entity_id: str, depth: int = 1) -> list[dict]:
        neighbors = []
        with self.driver.session() as session:
            # 出边
            result = session.run(
                "MATCH (a {id: $eid})-[r:RELATION]->(b) "
                "RETURN b, r.relation AS relation, r.detail AS detail",
                eid=entity_id,
            )
            for record in result:
                node_data = dict(record["b"])
                node_data["id"] = node_data.get("id", "")
                neighbors.append({
                    "entity": node_data,
                    "relation": record["relation"],
                    "detail": record["detail"],
                    "direction": "out",
                })

            # 入边
            result = session.run(
                "MATCH (a)-[r:RELATION]->(b {id: $eid}) "
                "RETURN a, r.relation AS relation, r.detail AS detail",
                eid=entity_id,
            )
            for record in result:
                node_data = dict(record["a"])
                node_data["id"] = node_data.get("id", "")
                neighbors.append({
                    "entity": node_data,
                    "relation": record["relation"],
                    "detail": record["detail"],
                    "direction": "in",
                })

        return neighbors

    def search_by_name(self, name: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n.name CONTAINS $name RETURN n",
                name=name,
            )
            entities = []
            for record in result:
                data = dict(record["n"])
                entities.append(data)
            return entities

    def find_path(self, source_id: str, target_id: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH p = shortestPath((a {id: $src})-[:RELATION*..5]-(b {id: $tgt})) "
                "RETURN p",
                src=source_id, tgt=target_id,
            )
            record = result.single()
            if not record:
                return []

            path = record["p"]
            result_list = []
            for i in range(len(path.nodes) - 1):
                result_list.append({
                    "from": dict(path.nodes[i]),
                    "to": dict(path.nodes[i + 1]),
                    "relation": path.relationships[i]["relation"],
                    "detail": path.relationships[i]["detail"],
                })
            return result_list

    def get_faction_members(self, faction_name: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Character {faction: $faction}) RETURN n",
                faction=faction_name,
            )
            return [dict(record["n"]) for record in result]

    def get_all_entities(self, entity_type: str = None) -> list[dict]:
        with self.driver.session() as session:
            if entity_type:
                # 首字母大写作为 label
                label = entity_type.capitalize()
                result = session.run(f"MATCH (n:{label}) RETURN n")
            else:
                result = session.run("MATCH (n) RETURN n")
            return [dict(record["n"]) for record in result]

    def get_stats(self) -> dict:
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edges = session.run(
                "MATCH ()-[r:RELATION]->() RETURN count(r) AS c"
            ).single()["c"]
            chars = session.run(
                "MATCH (n:Character) RETURN count(n) AS c"
            ).single()["c"]
            factions = session.run(
                "MATCH (n:Faction) RETURN count(n) AS c"
            ).single()["c"]
            events = session.run(
                "MATCH (n:Event) RETURN count(n) AS c"
            ).single()["c"]
            return {
                "nodes": nodes, "edges": edges,
                "characters": chars, "factions": factions, "events": events,
            }

    # ========== 图算法 ==========

    def detect_communities(self) -> dict:
        """社区发现：用 Neo4j GDS 库的 Louvain 算法"""
        with self.driver.session() as session:
            try:
                # 需要 Neo4j GDS 插件
                result = session.run(
                    "CALL gds.louvain.stream('graph') "
                    "YIELD nodeId, communityId "
                    "RETURN gds.util.asNode(nodeId).id AS id, "
                    "gds.util.asNode(nodeId).name AS name, communityId"
                )
                node_community = {}
                communities = {}
                for record in result:
                    nid = record["id"]
                    comm = record["communityId"]
                    node_community[nid] = comm
                    if comm not in communities:
                        communities[comm] = []
                    communities[comm].append({
                        "id": nid, "name": record["name"]
                    })

                community_list = [
                    {"index": k, "size": len(v), "members": v}
                    for k, v in sorted(communities.items())
                ]
                return {"node_map": node_community, "communities": community_list}
            except Exception:
                # GDS 未安装时回退到简单分组（按势力）
                return self._fallback_communities()

    def _fallback_communities(self) -> dict:
        """GDS 不可用时按势力分组"""
        node_community = {}
        communities = {}
        for entity in self.get_all_entities():
            faction = entity.get("faction", "其他")
            if faction not in communities:
                communities[faction] = []
            node_community[entity["id"]] = faction
            communities[faction].append(entity)

        community_list = [
            {"index": k, "size": len(v), "members": v}
            for k, v in communities.items()
        ]
        return {"node_map": node_community, "communities": community_list}

    def get_subgraph(self, entity_ids: list[str]) -> list[dict]:
        """提取子图"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (a)-[r:RELATION]->(b) "
                "WHERE a.id IN $ids AND b.id IN $ids "
                "RETURN a.id AS source, a.name AS source_name, "
                "b.id AS target, b.name AS target_name, "
                "r.relation AS relation, r.detail AS detail",
                ids=entity_ids,
            )
            return [dict(record) for record in result]

    def multi_hop_neighbors(self, entity_id: str, max_depth: int = 2) -> list[dict]:
        """多跳 BFS 遍历"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH p = (a {id: $eid})-[:RELATION*1.." + str(max_depth) + "]->(b) "
                "WHERE b.id <> $eid "
                "RETURN b, relationships(p) AS rels, length(p) AS depth",
                eid=entity_id,
            )
            visited = set()
            all_results = []
            for record in result:
                node_data = dict(record["b"])
                nid = node_data.get("id", "")
                if nid in visited:
                    continue
                visited.add(nid)

                rels = record["rels"]
                path_names = [entity_id]
                for r in rels:
                    path_names.append(f"--[{r['relation']}]-->")

                all_results.append({
                    "id": nid,
                    "entity": node_data,
                    "relation": rels[0]["relation"] if rels else "",
                    "detail": rels[0]["detail"] if rels else "",
                    "depth": record["depth"],
                    "path": path_names,
                })
            return all_results

    def relation_chain_query(self, entity_id: str, relation_types: list[str]) -> list[dict]:
        """关系推理链查询"""
        current_entities = [entity_id]
        chain_result = []

        for hop, rel_type in enumerate(relation_types):
            next_entities = []
            with self.driver.session() as session:
                for eid in current_entities:
                    result = session.run(
                        "MATCH (a {id: $eid})-[r:RELATION]->(b) "
                        "WHERE r.relation CONTAINS $rel "
                        "RETURN b, r.relation AS relation, r.detail AS detail",
                        eid=eid, rel=rel_type,
                    )
                    for record in result:
                        node_data = dict(record["b"])
                        nid = node_data.get("id", "")
                        if nid:
                            next_entities.append(nid)
                            chain_result.append({
                                "hop": hop + 1,
                                "relation": record["relation"],
                                "entity_id": nid,
                                "entity_name": node_data.get("name", ""),
                                "entity": node_data,
                                "detail": record["detail"],
                                "source_id": eid,
                            })
            current_entities = list(set(next_entities))
            if not current_entities:
                break

        return chain_result
