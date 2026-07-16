"""
查询理解与改写模块
1. 实体链接：别名/称号 → 标准实体名（"关二爷" → "关羽"）
2. 查询扩展：同义词 → 关系关键词（"对手" → "敌对"）
3. 查询意图分类：单跳 / 多跳 / 比较 / 描述
"""

import re
import json
from pathlib import Path


class QueryUnderstanding:
    """查询理解与改写"""

    # 别名/称号 → 标准实体名
    ALIAS_MAP = {
        # 蜀汉
        "玄德": "刘备", "皇叔": "刘备", "刘皇叔": "刘备", "先主": "刘备",
        "云长": "关羽", "关公": "关羽", "关二爷": "关羽", "武圣": "关羽", "美髯公": "关羽",
        "翼德": "张飞", "三爷": "张飞",
        "孔明": "诸葛亮", "卧龙": "诸葛亮", "诸葛孔明": "诸葛亮", "丞相": "诸葛亮",
        "子龙": "赵云", "常山赵子龙": "赵云",
        "孟起": "马超",
        "汉升": "黄忠", "老黄忠": "黄忠",
        "士元": "庞统", "凤雏": "庞统",
        "伯约": "姜维",
        "文长": "魏延",
        "幼常": "马谡",
        "元直": "徐庶",
        "孝直": "法正",
        "阿斗": "刘禅", "刘阿斗": "刘禅",
        # 曹魏
        "孟德": "曹操", "曹瞒": "曹操", "奸雄": "曹操",
        "元让": "夏侯惇",
        "妙才": "夏侯渊",
        "仲康": "许褚", "虎痴": "许褚",
        "奉孝": "郭嘉",
        "仲达": "司马懿", "司马仲达": "司马懿",
        "子桓": "曹丕",
        "子建": "曹植",
        "文远": "张辽",
        "文若": "荀彧",
        # 东吴
        "公瑾": "周瑜", "周郎": "周瑜",
        "子明": "吕蒙",
        "伯言": "陆逊",
        "子敬": "鲁肃",
        "公覆": "黄盖",
        "兴霸": "甘宁",
        "子义": "太史慈",
        "伯符": "孙策", "小霸王": "孙策",
        "文台": "孙坚",
        # 群雄
        "奉先": "吕布", "吕奉先": "吕布", "三姓家奴": "吕布",
        "仲颖": "董卓",
        "本初": "袁绍",
        "公路": "袁术",
        "伯圭": "公孙瓒",
    }

    # 同义词 → 关系类型（用于查询扩展和推理链匹配）
    SYNONYM_TO_RELATION = {
        "敌人": ["敌对", "曾为敌手", "被击败"],
        "敌手": ["敌对", "曾为敌手"],
        "对手": ["敌对", "曾为敌手"],
        "仇人": ["敌对"],
        "主公": ["君臣"],
        "君主": ["君臣"],
        "主子": ["君臣"],
        "部下": ["君臣"],
        "手下": ["君臣"],
        "部将": ["君臣"],
        "麾下": ["君臣"],
        "兄弟": ["结义兄弟"],
        "结义": ["结义兄弟"],
        "妻子": ["夫妻"],
        "夫人": ["夫妻"],
        "父亲": ["父子"],
        "儿子": ["父子"],
        "师父": ["师徒"],
        "徒弟": ["师徒"],
        "杀了": ["斩杀", "弑杀"],
        "斩了": ["斩杀"],
        "斩": ["斩杀"],
        "擒": ["擒杀"],
        "害死": ["斩杀", "擒杀"],
        "死": ["斩杀", "擒杀", "弑杀", "被杀"],
        "身亡": ["斩杀", "擒杀", "弑杀", "被杀"],
        "阵亡": ["斩杀", "被杀"],
        "击败": ["被击败", "击败"],
        "归降": ["归降", "投降"],
        "投降": ["归降", "投降"],
    }

    # 意图关键词
    INTENT_PATTERNS = {
        "multi_hop": [r".*的.*的.*", r".*的.*是谁", r".*的.*是什么"],
        "comparison": [r".*和.*[哪个谁比较]", r".*与.*[哪个谁比较]", r".*和.*.*区别"],
        "description": [r".*是谁", r".*是什么", r".*简介", r"介绍.*"],
    }

    def __init__(self, data_path: str = None):
        """初始化，可从 JSON 数据加载额外别名"""
        if data_path and Path(data_path).exists():
            self._load_aliases_from_data(data_path)

    def _load_aliases_from_data(self, data_path: str):
        """从知识图谱数据中提取字 → 标准名的映射"""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for char in data.get("entities", {}).get("characters", []):
            name = char["name"]
            desc = char.get("description", "")

            # 从描述中提取"字XX"
            m = re.search(r"字(\S+?)[，,。]", desc)
            if m:
                zi = m.group(1)
                if zi and zi not in self.ALIAS_MAP:
                    self.ALIAS_MAP[zi] = name

            # 从描述中提取"号XX"
            m = re.search(r"号(\S+?)[，,。]", desc)
            if m:
                hao = m.group(1)
                if hao and hao not in self.ALIAS_MAP:
                    self.ALIAS_MAP[hao] = name

    def entity_linking(self, query: str) -> tuple[str, list[dict]]:
        """
        实体链接：将查询中的别名替换为标准实体名
        返回：(改写后的查询, 替换记录列表)
        """
        rewritten = query
        replacements = []

        # 按别名长度降序匹配，避免"关公"被"公"先匹配
        sorted_aliases = sorted(self.ALIAS_MAP.items(), key=lambda x: len(x[0]), reverse=True)

        for alias, standard_name in sorted_aliases:
            if alias in rewritten:
                # 检查标准名是否已经在查询中
                if standard_name not in rewritten:
                    rewritten = rewritten.replace(alias, standard_name)
                    replacements.append({
                        "alias": alias,
                        "standard_name": standard_name,
                        "position": query.find(alias),
                    })

        return rewritten, replacements

    def query_expansion(self, query: str) -> list[str]:
        """
        查询扩展：将同义词扩展为多个关系关键词
        返回：扩展后的关键词列表（用于推理链匹配）
        """
        expanded_terms = []

        for synonym, relations in self.SYNONYM_TO_RELATION.items():
            if synonym in query:
                expanded_terms.extend(relations)

        return list(set(expanded_terms))

    def classify_intent(self, query: str) -> str:
        """
        查询意图分类
        返回：multi_hop / comparison / description / simple
        """
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, query):
                    return intent

        return "simple"

    def understand(self, query: str) -> dict:
        """
        完整查询理解流程
        返回：{
            "original_query": 原始查询,
            "rewritten_query": 改写后查询,
            "entity_replacements": 实体链接记录,
            "expanded_relations": 扩展的关系词,
            "intent": 查询意图,
        }
        """
        rewritten, replacements = self.entity_linking(query)
        expanded = self.query_expansion(query)
        intent = self.classify_intent(query)

        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "entity_replacements": replacements,
            "expanded_relations": expanded,
            "intent": intent,
        }


def demo():
    """演示查询理解"""
    qu = QueryUnderstanding("data/three_kingdoms.json")

    test_cases = [
        "关二爷的对手的主公是谁",
        "卧龙和仲达谁更厉害",
        "阿斗是谁",
        "虎痴杀了谁",
        "凤雏是怎么死的",
        "美髯公的敌人的主公",
    ]

    for q in test_cases:
        result = qu.understand(q)
        print(f"\n原始查询: {q}")
        print(f"  改写后: {result['rewritten_query']}")
        print(f"  意图: {result['intent']}")
        if result["entity_replacements"]:
            print(f"  实体链接:")
            for r in result["entity_replacements"]:
                print(f"    {r['alias']} → {r['standard_name']}")
        if result["expanded_relations"]:
            print(f"  关系扩展: {result['expanded_relations']}")


if __name__ == "__main__":
    demo()
