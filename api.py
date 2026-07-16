"""
三国演义知识图谱 RAG - Web API 服务
用 FastAPI 包装 RAGEngine，支持 HTTP 调用

启动方式：
  python api.py                # 默认端口 8000
  python api.py --port 9000    # 指定端口

或在 main.py 中：
  >>> serve 8000
"""
import sys
import argparse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from kg_builder import KnowledgeGraph
from rag_engine import RAGEngine


# ========== 全局引擎（启动时初始化一次，所有请求复用） ==========
_kg: KnowledgeGraph = None
_rag: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化引擎，关闭时清理"""
    global _kg, _rag
    print("[启动] 初始化知识图谱...")
    _kg = KnowledgeGraph("data/three_kingdoms.json")
    print(f"[启动] 图谱加载完成：{_kg.get_stats()}")

    print("[启动] 初始化 RAG 引擎...")
    try:
        _rag = RAGEngine(_kg)
        print("[启动] RAG 引擎就绪")
    except Exception as e:
        print(f"[启动] RAG 引擎初始化失败: {e}")
        print("[启动] 将仅支持图查询模式")

    yield  # 应用运行期间

    print("[关闭] 清理资源...")


app = FastAPI(
    title="三国演义知识图谱 RAG API",
    description="基于 GraphRAG 的三国演义智能问答系统",
    version="1.0.0",
    lifespan=lifespan,
)


# ========== 请求/响应模型 ==========

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AddEntityRequest(BaseModel):
    type: str  # character / faction / event
    id: str
    name: str
    faction: str = None
    description: str = ""


class AddRelationRequest(BaseModel):
    source: str
    target: str
    relation: str
    detail: str = ""


class RemoveRelationRequest(BaseModel):
    source: str
    target: str
    relation: str = None


# ========== API 接口 ==========

@app.get("/")
def api_info():
    """API 信息和可用端点"""
    return {
        "service": "三国演义知识图谱 RAG API",
        "version": "1.0.0",
        "endpoints": {
            "GET /stats": "图谱统计",
            "GET /search?name=关羽": "搜索实体",
            "GET /entity/{id}": "获取实体详情",
            "GET /entity/{id}/neighbors": "获取实体邻居",
            "POST /ask": "RAG 智能问答",
            "POST /entity": "添加实体",
            "POST /relation": "添加关系",
            "DELETE /entity/{id}": "删除实体",
            "DELETE /relation": "删除关系",
        },
    }


@app.get("/stats")
def stats():
    """图谱统计"""
    return _kg.get_stats()


@app.get("/search")
def search(name: str = Query(..., description="实体名称关键词")):
    """按名称搜索实体"""
    results = _kg.search_by_name(name)
    return {"query": name, "count": len(results), "results": results}


@app.get("/entity/{entity_id}")
def get_entity(entity_id: str):
    """获取单个实体详情"""
    entity = _kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"实体 '{entity_id}' 不存在")
    return {"id": entity_id, **entity}


@app.get("/entity/{entity_id}/neighbors")
def get_neighbors(entity_id: str):
    """获取实体的关系网络（邻居）"""
    if not _kg.get_entity(entity_id):
        raise HTTPException(status_code=404, detail=f"实体 '{entity_id}' 不存在")
    neighbors = _kg.get_neighbors(entity_id)
    return {"entity_id": entity_id, "count": len(neighbors), "neighbors": neighbors}


@app.post("/ask")
def ask(req: AskRequest):
    """
    RAG 智能问答
    返回完整的检索过程 + 综合回答
    """
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG 引擎不可用")

    result = _rag.query(req.question, top_k=req.top_k)
    answer = _rag.answer(req.question, top_k=req.top_k)

    return {
        "question": req.question,
        "answer": answer,
        "query_understanding": result.get("query_understanding", {}),
        "recall_hits": [
            {
                "id": h["id"],
                "name": h["metadata"]["name"],
                "rrf_score": h.get("rrf_score", 0),
            }
            for h in result.get("recall_hits", [])[:5]
        ],
        "reranked_hits": [
            {
                "id": h["id"],
                "name": h["metadata"]["name"],
                "rerank_score": h.get("rerank_score", 0),
                "entity_boost": h.get("entity_boost", False),
            }
            for h in result.get("reranked_hits", [])
        ],
        "graph_context_count": len(result.get("graph_context", [])),
        "subgraph_context": result.get("subgraph_context", []),
        "chain_context": result.get("chain_context", []),
    }


@app.post("/entity")
def add_entity(req: AddEntityRequest):
    """添加实体（自动保存+重建索引）"""
    if req.type not in ("character", "faction", "event"):
        raise HTTPException(status_code=400, detail="type 必须是 character/faction/event")

    success = _kg.add_entity(req.type, req.id, req.name, req.description, req.faction)
    if not success:
        raise HTTPException(status_code=409, detail=f"实体 '{req.id}' 已存在")

    _kg.save_to_file()
    if _rag:
        _rag.rebuild_index()

    return {"message": f"已添加实体: {req.name}（{req.id}）", "stats": _kg.get_stats()}


@app.post("/relation")
def add_relation(req: AddRelationRequest):
    """添加关系（自动保存+重建索引）"""
    success = _kg.add_relation(req.source, req.target, req.relation, req.detail)
    if not success:
        if req.source not in _kg.graph.nodes:
            raise HTTPException(status_code=404, detail=f"源实体 '{req.source}' 不存在")
        elif req.target not in _kg.graph.nodes:
            raise HTTPException(status_code=404, detail=f"目标实体 '{req.target}' 不存在")
        else:
            raise HTTPException(status_code=409, detail="关系已存在")

    _kg.save_to_file()
    if _rag:
        _rag.rebuild_index()

    return {"message": f"已添加关系: {req.source} --[{req.relation}]--> {req.target}"}


@app.delete("/entity/{entity_id}")
def remove_entity(entity_id: str):
    """删除实体（自动保存+重建索引）"""
    entity = _kg.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"实体 '{entity_id}' 不存在")

    name = entity.get("name", entity_id)
    _kg.remove_entity(entity_id)
    _kg.save_to_file()
    if _rag:
        _rag.rebuild_index()

    return {"message": f"已删除实体: {name}（{entity_id}）", "stats": _kg.get_stats()}


@app.delete("/relation")
def remove_relation(req: RemoveRelationRequest):
    """删除关系（自动保存+重建索引）"""
    removed = _kg.remove_relation(req.source, req.target, req.relation)
    if removed == 0:
        raise HTTPException(status_code=404, detail="未找到匹配的关系")

    _kg.save_to_file()
    if _rag:
        _rag.rebuild_index()

    return {"message": f"已删除 {removed} 条关系", "removed_count": removed}


# ========== 启动入口 ==========

def main():
    parser = argparse.ArgumentParser(description="三国演义知识图谱 RAG API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="监听端口（默认 8000）")
    args = parser.parse_args()

    import uvicorn
    print(f"\n三国演义知识图谱 RAG API")
    print(f"启动后访问: http://localhost:{args.port}")
    print(f"API 文档: http://localhost:{args.port}/docs")
    print(f"按 Ctrl+C 停止\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
