"""
阶段三：自动实体抽取与关系建图
从原始文本中自动抽取实体和关系，替代手工编辑 JSON

策略：
1. 实体识别：基于已有实体词典 + 规则匹配（无需 LLM）
2. 关系抽取：基于共现窗口 + 关系关键词模板
3. 输出：与 three_kingdoms.json 相同格式的 JSON

使用方式：
    from entity_extractor import EntityExtractor
    extractor = EntityExtractor()
    result = extractor.extract_from_text(raw_text)
"""

import re
import json
from pathlib import Path
from collections import defaultdict


class EntityExtractor:
    """自动实体抽取器"""

    # 关系关键词模板：关键词 → 关系类型
    RELATION_PATTERNS = {
        "结义兄弟": [r"结义", r"桃园.{0,4}结", r"三兄弟"],
        "君臣": [r"效忠", r"归降", r"投靠", r"部下", r"麾下", r"主公", r"臣子"],
        "父子": [r"之子", r"长子", r"次子", r"养子", r"义子", r"父亲"],
        "夫妻": [r"妻", r"夫人", r"嫁", r"娶"],
        "师徒": [r"师父", r"徒弟", r"拜师", r"学艺"],
        "敌对": [r"敌", r"仇", r"对战", r"交锋", r"对抗"],
        "斩杀": [r"斩", r"杀", r"擒杀", r"刺杀", r"处死"],
        "联盟": [r"联盟", r"联合", r"结盟", r"合作"],
        "归降": [r"投降", r"归降", r"归顺", r"降"],
    }

    # 人物称谓后缀（用于辅助识别人物名）
    NAME_SUFFIXES = ["公", "侯", "将军", "帝", "王", "主", "相", "督"]

    def __init__(self, existing_entities_path: str = None):
        """
        初始化抽取器
        existing_entities_path: 已有实体词典路径（用于增强识别）
        """
        self.entity_dict = {}  # name → {id, faction, type}
        self.alias_map = {}    # 别名 → 标准名

        if existing_entities_path:
            self._load_existing_entities(existing_entities_path)

    def _load_existing_entities(self, path: str):
        """加载已有实体词典"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for char in data.get("entities", {}).get("characters", []):
            self.entity_dict[char["name"]] = {
                "id": char["id"],
                "faction": char.get("faction", ""),
                "type": "character",
            }
            # 加载别名
            for alias in char.get("aliases", []):
                self.alias_map[alias] = char["name"]

    def extract_entities(self, text: str) -> list[dict]:
        """
        从文本中识别实体（人物名）
        策略：先用已有词典匹配，再用规则发现新实体
        """
        found = {}

        # 1. 词典匹配
        for name, info in self.entity_dict.items():
            if name in text:
                if name not in found:
                    found[name] = info

        # 2. 别名匹配
        for alias, standard_name in self.alias_map.items():
            if alias in text and standard_name in self.entity_dict:
                if standard_name not in found:
                    found[standard_name] = self.entity_dict[standard_name]

        # 3. 规则发现新实体（2-3字中文人名 + 称谓后缀）
        # 匹配 "X公"、"X将军" 等模式
        pattern = r"([\u4e00-\u9fa5]{2,3})(?:" + "|".join(self.NAME_SUFFIXES) + r")"
        matches = re.findall(pattern, text)
        for name in matches:
            if name not in found and len(name) >= 2:
                found[name] = {
                    "id": self._name_to_id(name),
                    "faction": "",
                    "type": "character",
                    "auto_extracted": True,
                }

        return [
            {"name": name, **info}
            for name, info in found.items()
        ]

    def extract_relations(self, text: str, entities: list[dict] = None) -> list[dict]:
        """
        从文本中抽取实体间关系
        策略：在共现窗口内检测关系关键词
        """
        if entities is None:
            entities = self.extract_entities(text)

        if len(entities) < 2:
            return []

        relations = []
        entity_names = [e["name"] for e in entities]

        # 对每对共现实体，在窗口内找关系关键词
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i >= j:
                    continue

                # 找到两个实体在文本中的位置
                positions1 = self._find_all(text, e1["name"])
                positions2 = self._find_all(text, e2["name"])

                for p1 in positions1:
                    for p2 in positions2:
                        # 共现窗口：两个实体之间的距离 < 50 字符
                        distance = abs(p1 - p2)
                        if distance > 50:
                            continue

                        # 提取两个实体之间的文本
                        start = min(p1, p2) + len(e1["name"])
                        end = max(p1, p2)
                        between_text = text[start:end]

                        # 检测关系类型
                        for rel_type, patterns in self.RELATION_PATTERNS.items():
                            for pattern in patterns:
                                if re.search(pattern, between_text):
                                    relations.append({
                                        "source": e1["id"],
                                        "target": e2["id"],
                                        "relation": rel_type,
                                        "detail": f"自动抽取：{between_text.strip()[:30]}",
                                        "auto_extracted": True,
                                    })
                                    break
                            else:
                                continue
                            break

                        # 如果没有关键词但共现，标记为"关联"
                        if not any(
                            r["source"] == e1["id"] and r["target"] == e2["id"]
                            for r in relations
                        ):
                            relations.append({
                                "source": e1["id"],
                                "target": e2["id"],
                                "relation": "关联",
                                "detail": "自动抽取：共现",
                                "auto_extracted": True,
                            })

        return relations

    def extract_from_text(self, text: str) -> dict:
        """
        完整抽取流程：实体 → 关系 → 输出 JSON
        """
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)

        # 按 type 分组
        characters = [e for e in entities if e.get("type") == "character"]
        events = self._extract_events(text)

        return {
            "entities": {
                "characters": [
                    {
                        "id": e["id"],
                        "name": e["name"],
                        "faction": e.get("faction", ""),
                        "description": e.get("description", "自动抽取"),
                    }
                    for e in characters
                ],
                "factions": [],
                "events": events,
            },
            "relationships": relations,
            "metadata": {
                "source": "auto_extracted",
                "entity_count": len(entities),
                "relation_count": len(relations),
            },
        }

    def _extract_events(self, text: str) -> list[dict]:
        """简单事件抽取：识别"XX之战"模式"""
        events = []
        pattern = r"([\u4e00-\u9fa5]{2,4})之战"
        matches = re.finditer(pattern, text)
        seen = set()
        for m in matches:
            event_name = m.group()
            if event_name not in seen:
                seen.add(event_name)
                events.append({
                    "id": self._name_to_id(event_name),
                    "name": event_name,
                    "description": f"自动抽取事件：{event_name}",
                })
        return events

    def _find_all(self, text: str, substring: str) -> list[int]:
        """查找所有出现位置"""
        positions = []
        start = 0
        while True:
            pos = text.find(substring, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions

    def _name_to_id(self, name: str) -> str:
        """将中文名转为 ID（拼音首字母简化版）"""
        # 简单方案：用字符的 unicode 编码
        return "auto_" + str(abs(hash(name)) % 100000)


def demo():
    """演示自动抽取"""
    sample_text = """
    关羽与刘备、张飞在桃园结义，成为结义兄弟。关羽效忠刘备，为蜀汉五虎将之首。
    赤壁之战中，诸葛亮联合周瑜对抗曹操。诸葛亮借东风，周瑜火攻曹军。
    吕布被董卓收买，杀丁原后认董卓为义父。后因貂蝉，吕布杀董卓。
    关羽在白马之战斩颜良，在襄樊之战水淹七军擒于禁、斩庞德。
    """

    extractor = EntityExtractor("data/three_kingdoms.json")
    result = extractor.extract_from_text(sample_text)

    print(f"抽取到 {len(result['entities']['characters'])} 个实体")
    print(f"抽取到 {len(result['relationships'])} 条关系")
    print(f"抽取到 {len(result['entities']['events'])} 个事件")
    print()

    print("=== 实体 ===")
    for e in result["entities"]["characters"]:
        print(f"  {e['name']} ({e.get('faction', '未知')})")

    print("\n=== 关系 ===")
    for r in result["relationships"]:
        print(f"  {r['source']} --[{r['relation']}]--> {r['target']}：{r['detail']}")


if __name__ == "__main__":
    demo()
