"""
RAG 评估模块
评估指标：
  1. Recall@K：Top-K 检索结果中是否包含正确实体
  2. MRR（Mean Reciprocal Rank）：第一个正确实体的排名倒数的平均值
  3. NDCG@K：归一化折损累积增益，考虑位置加权
  4. Answer Accuracy：最终回答中是否包含正确答案关键词
"""

import json
import math
import re
from pathlib import Path


class Evaluator:
    """RAG 系统评估器"""

    def __init__(self, dataset_path: str = "data/eval_dataset.json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        self.test_cases = self.dataset["test_cases"]

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
        """
        Recall@K：Top-K 中命中了多少个正确实体
        返回值：[0, 1]，1 表示全部命中
        """
        top_k = retrieved_ids[:k]
        hits = sum(1 for eid in expected_ids if eid in top_k)
        return hits / len(expected_ids) if expected_ids else 0.0

    @staticmethod
    def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
        """
        MRR（Mean Reciprocal Rank）：第一个正确实体的排名倒数
        返回值：[0, 1]，1 表示第 1 名就命中
        """
        for i, rid in enumerate(retrieved_ids):
            if rid in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
        """
        NDCG@K：归一化折损累积增益
        正确实体排越靠前分数越高，用 DCG/IDCG 归一化
        返回值：[0, 1]，1 表示完美排序
        """
        # DCG：每个位置的贡献 = 1/log2(rank+1)，只有命中的才有贡献
        dcg = 0.0
        for i, rid in enumerate(retrieved_ids[:k]):
            if rid in expected_ids:
                dcg += 1.0 / math.log2(i + 2)  # i+2 因为 rank 从 1 开始，log2(1)=0 无意义

        # IDCG：理想情况下所有正确实体都排在最前面
        ideal_hits = min(len(expected_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def answer_accuracy(answer: str, expected_answer: str) -> float:
        """
        回答准确率：检查回答中是否包含正确答案的关键词
        返回值：0.0（完全不包含）~ 1.0（全部包含）
        """
        if not expected_answer:
            return 0.0

        # 提取答案关键词（按顿号、逗号分隔）
        keywords = re.split(r"[、，,。；;（）()]", expected_answer)
        keywords = [k.strip() for k in keywords if k.strip() and len(k.strip()) >= 2]

        if not keywords:
            return 1.0 if expected_answer in answer else 0.0

        hits = sum(1 for kw in keywords if kw in answer)
        return hits / len(keywords)

    def evaluate(self, rag_engine, verbose: bool = True) -> dict:
        """
        对 RAG 引擎进行全面评估
        """
        results = []

        for tc in self.test_cases:
            question = tc["question"]
            expected_ids = tc["expected_entities"]
            expected_answer = tc["expected_answer"]

            if verbose:
                print(f"\n[{tc['id']}/{len(self.test_cases)}] {question}")
                print(f"  期望实体: {expected_ids}")

            # 执行查询
            result = rag_engine.query(question)
            retrieved_ids = [h["id"] for h in result["reranked_hits"]]

            # 计算检索指标
            recall_5 = self.recall_at_k(retrieved_ids, expected_ids, k=5)
            recall_3 = self.recall_at_k(retrieved_ids, expected_ids, k=3)
            mrr_score = self.mrr(retrieved_ids, expected_ids)
            ndcg_5 = self.ndcg_at_k(retrieved_ids, expected_ids, k=5)

            # 计算回答准确率
            answer = rag_engine.answer(question)
            acc = self.answer_accuracy(answer, expected_answer)

            # 找到命中的实体名
            hit_names = []
            for h in result["reranked_hits"]:
                if h["id"] in expected_ids:
                    hit_names.append(h["metadata"]["name"])

            if verbose:
                retrieved_names = [h["metadata"]["name"] for h in result["reranked_hits"][:5]]
                print(f"  检索Top5: {retrieved_names}")
                print(f"  命中: {hit_names if hit_names else '无'}")
                print(f"  Recall@3={recall_3:.2f} Recall@5={recall_5:.2f} MRR={mrr_score:.2f} NDCG@5={ndcg_5:.2f}")
                print(f"  回答准确率={acc:.2f}")

            results.append({
                "id": tc["id"],
                "question": question,
                "category": tc["category"],
                "difficulty": tc["difficulty"],
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids[:5],
                "hit_names": hit_names,
                "recall_at_3": recall_3,
                "recall_at_5": recall_5,
                "mrr": mrr_score,
                "ndcg_at_5": ndcg_5,
                "answer_accuracy": acc,
                "answer": answer[:200],
            })

        # 汇总指标
        n = len(results)
        summary = {
            "total": n,
            "recall_at_3": sum(r["recall_at_3"] for r in results) / n,
            "recall_at_5": sum(r["recall_at_5"] for r in results) / n,
            "mrr": sum(r["mrr"] for r in results) / n,
            "ndcg_at_5": sum(r["ndcg_at_5"] for r in results) / n,
            "answer_accuracy": sum(r["answer_accuracy"] for r in results) / n,
        }

        # 按类别分组统计
        by_category = {}
        for r in results:
            cat = r["category"]
            if cat not in by_category:
                by_category[cat] = {"count": 0, "recall": 0, "mrr": 0, "ndcg": 0, "acc": 0}
            c = by_category[cat]
            c["count"] += 1
            c["recall"] += r["recall_at_5"]
            c["mrr"] += r["mrr"]
            c["ndcg"] += r["ndcg_at_5"]
            c["acc"] += r["answer_accuracy"]

        for cat, c in by_category.items():
            c["recall"] /= c["count"]
            c["mrr"] /= c["count"]
            c["ndcg"] /= c["count"]
            c["acc"] /= c["count"]

        summary["by_category"] = by_category

        # 按难度分组统计
        by_difficulty = {}
        for r in results:
            diff = r["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = {"count": 0, "recall": 0, "mrr": 0, "ndcg": 0, "acc": 0}
            d = by_difficulty[diff]
            d["count"] += 1
            d["recall"] += r["recall_at_5"]
            d["mrr"] += r["mrr"]
            d["ndcg"] += r["ndcg_at_5"]
            d["acc"] += r["answer_accuracy"]

        for diff, d in by_difficulty.items():
            d["recall"] /= d["count"]
            d["mrr"] /= d["count"]
            d["ndcg"] /= d["count"]
            d["acc"] /= d["count"]

        summary["by_difficulty"] = by_difficulty

        return {"summary": summary, "details": results}

    @staticmethod
    def print_report(eval_result: dict):
        """打印评估报告"""
        s = eval_result["summary"]

        print("\n" + "=" * 60)
        print("              RAG 系统评估报告")
        print("=" * 60)

        print(f"\n测试用例数: {s['total']}")

        print(f"\n{'指标':<20} {'分数':<10} {'说明'}")
        print("-" * 55)
        print(f"{'Recall@3':<20} {s['recall_at_3']:<10.4f} Top-3 命中率")
        print(f"{'Recall@5':<20} {s['recall_at_5']:<10.4f} Top-5 命中率")
        print(f"{'MRR':<20} {s['mrr']:<10.4f} 平均倒数排名")
        print(f"{'NDCG@5':<20} {s['ndcg_at_5']:<10.4f} 归一化折损累积增益")
        print(f"{'Answer Accuracy':<20} {s['answer_accuracy']:<10.4f} 回答准确率")

        print(f"\n{'按类别分组':}")
        print(f"{'类别':<15} {'数量':<6} {'Recall@5':<10} {'MRR':<10} {'NDCG@5':<10} {'准确率':<10}")
        print("-" * 60)
        for cat, c in s["by_category"].items():
            print(f"{cat:<15} {c['count']:<6} {c['recall']:<10.4f} {c['mrr']:<10.4f} {c['ndcg']:<10.4f} {c['acc']:<10.4f}")

        print(f"\n{'按难度分组':}")
        print(f"{'难度':<15} {'数量':<6} {'Recall@5':<10} {'MRR':<10} {'NDCG@5':<10} {'准确率':<10}")
        print("-" * 60)
        for diff, d in s["by_difficulty"].items():
            print(f"{diff:<15} {d['count']:<6} {d['recall']:<10.4f} {d['mrr']:<10.4f} {d['ndcg']:<10.4f} {d['acc']:<10.4f}")

        # 失败案例
        failed = [r for r in eval_result["details"] if r["recall_at_5"] < 0.5]
        if failed:
            print(f"\n{'失败案例 (Recall@5 < 0.5)'}:")
            print("-" * 60)
            for r in failed:
                print(f"  #{r['id']} [{r['difficulty']}] {r['question']}")
                print(f"    期望: {r['expected_ids']}")
                print(f"    检索: {r['retrieved_ids']}")
                print(f"    命中: {r['hit_names'] if r['hit_names'] else '无'}")

        print("\n" + "=" * 60)

    def evaluate_and_report(self, rag_engine, verbose: bool = True) -> dict:
        """评估并打印报告"""
        result = self.evaluate(rag_engine, verbose=verbose)
        self.print_report(result)
        return result


if __name__ == "__main__":
    from kg_builder import KnowledgeGraph
    from rag_engine import RAGEngine

    kg = KnowledgeGraph("data/three_kingdoms.json")
    rag = RAGEngine(kg)

    evaluator = Evaluator()
    evaluator.evaluate_and_report(rag, verbose=True)
