"""
三国演义知识图谱 RAG - 交互式查询终端
"""
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from kg_builder import KnowledgeGraph
from rag_engine import RAGEngine

console = Console()


def print_banner():
    banner = """
╔══════════════════════════════════════════════╗
║      🏯 三国演义 · 知识图谱 RAG 系统        ║
║      Knowledge Graph RAG for                 ║
║      Romance of the Three Kingdoms           ║
╚══════════════════════════════════════════════╝
"""
    console.print(banner, style="bold yellow")


def show_stats(kg: KnowledgeGraph):
    stats = kg.get_stats()
    table = Table(title="知识图谱统计")
    table.add_column("指标", style="cyan")
    table.add_column("数量", style="green")
    table.add_row("总节点", str(stats["nodes"]))
    table.add_row("总关系边", str(stats["edges"]))
    table.add_row("人物", str(stats["characters"]))
    table.add_row("势力", str(stats["factions"]))
    table.add_row("事件", str(stats["events"]))
    console.print(table)


def cmd_search(kg: KnowledgeGraph, name: str):
    results = kg.search_by_name(name)
    if not results:
        console.print(f"[red]未找到与 '{name}' 相关的实体[/red]")
        return

    table = Table(title=f"搜索结果：{name}")
    table.add_column("ID", style="dim")
    table.add_column("名称", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("势力", style="yellow")
    table.add_column("简介", style="white")

    for r in results:
        table.add_row(
            r["id"],
            r["name"],
            r.get("type", ""),
            r.get("faction", ""),
            r.get("description", "")[:60] + "...",
        )
    console.print(table)


def cmd_neighbors(kg: KnowledgeGraph, entity_id: str):
    entity = kg.get_entity(entity_id)
    if not entity:
        console.print(f"[red]实体 '{entity_id}' 不存在[/red]")
        return

    console.print(f"\n[bold cyan]{entity['name']}[/bold cyan] 的关系网络：\n")

    neighbors = kg.get_neighbors(entity_id)
    if not neighbors:
        console.print("[yellow]该实体没有关联关系[/yellow]")
        return

    table = Table()
    table.add_column("关系", style="magenta")
    table.add_column("关联实体", style="cyan")
    table.add_column("详情", style="white")
    table.add_column("方向", style="dim")

    for n in neighbors:
        direction = "→" if n["direction"] == "out" else "←"
        table.add_row(
            n["relation"],
            n["entity"]["name"],
            n["detail"],
            direction,
        )
    console.print(table)


def cmd_path(kg: KnowledgeGraph, source: str, target: str):
    path = kg.find_path(source, target)
    if not path:
        console.print(f"[yellow]'{source}' 和 '{target}' 之间没有路径[/yellow]")
        return

    console.print(f"\n[bold]从 {path[0]['from']['name']} 到 {path[-1]['to']['name']} 的路径：[/bold]\n")
    for i, step in enumerate(path):
        console.print(
            f"  {i + 1}. [{step['from']['name']}] "
            f"[magenta]--{step['relation']}-->[/magenta] "
            f"[{step['to']['name']}]"
        )
        console.print(f"     {step['detail']}", style="dim")


def cmd_faction(kg: KnowledgeGraph, faction_name: str):
    members = kg.get_faction_members(faction_name)
    if not members:
        console.print(f"[red]未找到势力 '{faction_name}'[/red]")
        return

    console.print(f"\n[bold cyan]{faction_name}[/bold cyan] 的成员：\n")
    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("名称", style="cyan")
    table.add_column("简介", style="white")

    for m in members:
        table.add_row(m["id"], m["name"], m.get("description", "")[:60] + "...")
    console.print(table)


def cmd_rag(rag: RAGEngine, question: str):
    console.print(f"\n[bold]问题：[/bold]{question}\n")
    console.print("[dim]正在检索（查询理解 → 混合召回 → 重排序 → 社区级 → 多跳图 → 子图 → 推理链）...[/dim]\n")

    result = rag.query(question)

    # 查询理解结果
    qu = result.get("query_understanding", {})
    if qu.get("entity_replacements") or qu.get("expanded_relations"):
        console.print(Panel.fit("[bold magenta]查询理解[/bold magenta]"))
        if qu["rewritten_query"] != qu["original_query"]:
            console.print(f"  改写: [dim]{qu['original_query']}[/dim] → [cyan]{qu['rewritten_query']}[/cyan]")
        if qu.get("entity_replacements"):
            for r in qu["entity_replacements"]:
                console.print(f"  实体链接: [yellow]{r['alias']}[/yellow] → [cyan]{r['standard_name']}[/cyan]")
        if qu.get("expanded_relations"):
            console.print(f"  关系扩展: {', '.join(qu['expanded_relations'])}")
        console.print(f"  意图: {qu.get('intent', 'simple')}\n")

    # 混合召回结果
    console.print(Panel.fit("[bold green]混合召回（BM25 + 向量 RRF 融合）[/bold green]"))
    for i, h in enumerate(result["recall_hits"][:5]):
        bm25_mark = f"BM25#{h.get('bm25_rank', '-')}" if "bm25_rank" in h else "BM25:无"
        vec_mark = f"Vec#{h.get('vector_rank', '-')}" if "vector_rank" in h else "Vec:无"
        console.print(
            f"  {i + 1}. [cyan]{h['metadata']['name']}[/cyan] "
            f"({bm25_mark}, {vec_mark}, RRF={h.get('rrf_score', 0):.4f})"
        )

    # 重排序结果
    console.print(Panel.fit("[bold magenta]重排序后[/bold magenta]"))
    for i, h in enumerate(result["reranked_hits"]):
        boost_mark = " [green](实体加分)[/green]" if h.get("entity_boost") else ""
        console.print(
            f"  {i + 1}. [cyan]{h['metadata']['name']}[/cyan] "
            f"(Rerank={h.get('rerank_score', 0):.4f}){boost_mark}"
        )

    # 多跳图扩展（带路径）
    if result.get("graph_context"):
        console.print(Panel.fit("[bold blue]多跳图扩展（深度2，带关系路径）[/bold blue]"))
        for g in result["graph_context"][:8]:
            path_str = " ".join(g.get("path", []))
            console.print(f"  [dim]跳{g['depth']}[/dim] {path_str}")

    # 命中实体间的直接关系（子图）
    if result.get("subgraph_context"):
        console.print(Panel.fit("[bold cyan]命中实体间直接关系（子图）[/bold cyan]"))
        for e in result["subgraph_context"]:
            console.print(
                f"  • {e['source_name']} --[{e['relation']}]--> {e['target_name']}"
            )

    # 关系推理链
    if result.get("chain_context"):
        console.print(Panel.fit("[bold red]关系推理链[/bold red]"))
        for c in result["chain_context"]:
            console.print(
                f"  [dim]跳{c['hop']}[/dim] {c['source_name']} --[{c['relation']}]--> "
                f"[cyan]{c['entity_name']}[/cyan]：{c['detail']}"
            )

    # 社区级检索结果（阶段三：分层检索）
    if result.get("community_search_results"):
        console.print(Panel.fit("[bold yellow]社区级检索（分层检索 Top-3）[/bold yellow]"))
        for cr in result["community_search_results"]:
            console.print(
                f"  社区#{cr['community_index']} (Score={cr['score']:.4f})：{', '.join(cr['members'][:6])}"
            )

    # 社区上下文
    if result.get("community_context"):
        console.print(Panel.fit("[bold yellow]命中实体所在社区[/bold yellow]"))
        for c in result["community_context"]:
            console.print(
                f"  社区#{c['community_index']}（{c['size']}人）：{', '.join(c['members'][:8])}"
            )

    # 综合回答
    console.print(Panel.fit("[bold yellow]综合回答[/bold yellow]"))
    answer = rag.answer(question)
    console.print(answer)


def print_help():
    help_text = """
[bold]可用命令：[/bold]

  [cyan]search <名称>[/cyan]      - 按名称搜索实体
  [cyan]neighbors <ID>[/cyan]     - 查看实体的关系网络
  [cyan]path <源ID> <目标ID>[/cyan] - 查找两个实体间的关系路径
  [cyan]faction <势力名>[/cyan]   - 查看势力成员（蜀汉/曹魏/东吴/群雄）
  [cyan]ask <问题>[/cyan]         - RAG 智能问答（三路融合检索）
  [cyan]community[/cyan]          - 查看社区发现结果+摘要
  [cyan]extract <文本>[/cyan]     - 从文本自动抽取实体和关系
  [cyan]eval[/cyan]               - 运行评估（22 个用例，输出 Recall/MRR/NDCG/准确率）
  [cyan]add_entity <type> <id> <name> <faction> <desc>[/cyan] - 添加实体
  [cyan]add_relation <src> <tgt> <relation> <detail>[/cyan]   - 添加关系
  [cyan]remove_entity <id>[/cyan] - 删除实体（及其所有关联关系）
  [cyan]remove_relation <src> <tgt> [relation][/cyan] - 删除关系
  [cyan]serve [port][/cyan]      - 启动 Web API 服务（默认端口 8000）
  [cyan]stats[/cyan]              - 查看图谱统计
  [cyan]list <类型>[/cyan]        - 列出所有实体（character/faction/event）
  [cyan]viz [html|png][/cyan]     - 可视化知识图谱（html=交互式，png=静态图）
  [cyan]help[/cyan]               - 显示帮助
  [cyan]quit[/cyan]               - 退出

[bold]示例：[/bold]
  search 关羽
  neighbors guanyu
  ask 诸葛亮和司马懿是什么关系？
  add_entity character simayi 司马懿 曹魏 字仲达，司马家族成员
  add_relation guanyi lvbu 敌对 虎牢关三英战吕布
  remove_entity simayi
  remove_relation guanyu lvbu 敌对
  list character
"""
    console.print(Panel(help_text, title="帮助"))


def cmd_community(rag: RAGEngine, kg: KnowledgeGraph):
    """显示社区发现结果和摘要（阶段三）"""
    if rag is None:
        console.print("[red]RAG 引擎不可用，无法显示社区摘要[/red]")
        return

    communities = rag.community_summaries
    console.print(f"\n[bold]共检测到 {len(communities)} 个社区[/bold]\n")

    for comm in communities:
        console.print(
            Panel.fit(
                f"[bold]社区#{comm['index']}[/bold]（{comm['size']}人）\n"
                f"成员：{', '.join(comm['members'])}\n"
                f"内部关系（{len(comm['relations'])}条）：\n"
                + "\n".join(f"  {r}" for r in comm["relations"][:8])
                + ("\n  ..." if len(comm["relations"]) > 8 else ""),
                title=f"社区#{comm['index']}",
            )
        )


def cmd_extract(text: str):
    """从文本自动抽取实体和关系（阶段三）"""
    from entity_extractor import EntityExtractor

    console.print(f"\n[bold]输入文本：[/bold]{text}\n")
    console.print("[dim]正在抽取实体和关系...[/dim]\n")

    extractor = EntityExtractor("data/three_kingdoms.json")
    result = extractor.extract_from_text(text)

    console.print(f"[green]抽取到 {len(result['entities']['characters'])} 个实体[/green]")
    for e in result["entities"]["characters"]:
        faction = e.get("faction", "未知")
        auto = " (新发现)" if e.get("auto_extracted") else ""
        console.print(f"  • {e['name']}（{faction}）{auto}")

    console.print(f"\n[green]抽取到 {len(result['relationships'])} 条关系[/green]")
    for r in result["relationships"]:
        console.print(
            f"  • {r['source']} --[{r['relation']}]--> {r['target']}：{r['detail']}"
        )

    if result["entities"]["events"]:
        console.print(f"\n[green]抽取到 {len(result['entities']['events'])} 个事件[/green]")
        for e in result["entities"]["events"]:
            console.print(f"  • {e['name']}")


def cmd_add_entity(kg: KnowledgeGraph, rag: RAGEngine, args: str):
    """添加实体：add_entity <type> <id> <name> <faction> <description...>"""
    parts = args.split(maxsplit=4)
    if len(parts) < 4:
        console.print("[red]用法: add_entity <type> <id> <name> <faction> <description...>[/red]")
        console.print("[dim]  type: character/faction/event[/dim]")
        console.print("[dim]  faction: 势力名（非 character 类型用 - 占位）[/dim]")
        console.print("[dim]示例: add_entity character simayi 司马懿 曹魏 字仲达，司马家族成员[/dim]")
        return

    etype = parts[0]
    entity_id = parts[1]
    name = parts[2]
    faction = parts[3] if parts[3] != "-" else None
    description = parts[4] if len(parts) > 4 else ""

    if etype not in ("character", "faction", "event"):
        console.print(f"[red]未知类型: {etype}，可选: character/faction/event[/red]")
        return

    success = kg.add_entity(etype, entity_id, name, description, faction)
    if not success:
        console.print(f"[red]实体 '{entity_id}' 已存在[/red]")
        return

    kg.save_to_file()
    console.print(f"[green]已添加实体: {name}（{entity_id}）[/green]")

    if rag:
        console.print("[dim]正在重建索引...[/dim]")
        rag.rebuild_index()
        console.print("[green]索引重建完成，可立即查询[/green]")


def cmd_add_relation(kg: KnowledgeGraph, rag: RAGEngine, args: str):
    """添加关系：add_relation <source_id> <target_id> <relation> <detail...>"""
    parts = args.split(maxsplit=3)
    if len(parts) < 3:
        console.print("[red]用法: add_relation <source_id> <target_id> <relation> <detail...>[/red]")
        console.print("[dim]示例: add_relation guanyu lvbu 敌对 虎牢关三英战吕布[/dim]")
        return

    source = parts[0]
    target = parts[1]
    relation = parts[2]
    detail = parts[3] if len(parts) > 3 else ""

    success = kg.add_relation(source, target, relation, detail)
    if not success:
        if source not in kg.graph.nodes:
            console.print(f"[red]源实体 '{source}' 不存在[/red]")
        elif target not in kg.graph.nodes:
            console.print(f"[red]目标实体 '{target}' 不存在[/red]")
        else:
            console.print(f"[red]关系已存在: {source} --[{relation}]--> {target}[/red]")
        return

    kg.save_to_file()
    console.print(f"[green]已添加关系: {source} --[{relation}]--> {target}[/green]")

    if rag:
        console.print("[dim]正在重建索引...[/dim]")
        rag.rebuild_index()
        console.print("[green]索引重建完成，可立即查询[/green]")


def cmd_remove_entity(kg: KnowledgeGraph, rag: RAGEngine, args: str):
    """删除实体：remove_entity <id>"""
    entity_id = args.strip()
    if not entity_id:
        console.print("[red]用法: remove_entity <实体ID>[/red]")
        return

    entity = kg.get_entity(entity_id)
    if not entity:
        console.print(f"[red]实体 '{entity_id}' 不存在[/red]")
        return

    name = entity.get("name", entity_id)
    success = kg.remove_entity(entity_id)
    if not success:
        console.print(f"[red]删除失败[/red]")
        return

    kg.save_to_file()
    console.print(f"[green]已删除实体: {name}（{entity_id}）及所有关联关系[/green]")

    if rag:
        console.print("[dim]正在重建索引...[/dim]")
        rag.rebuild_index()
        console.print("[green]索引重建完成[/green]")


def cmd_remove_relation(kg: KnowledgeGraph, rag: RAGEngine, args: str):
    """删除关系：remove_relation <source_id> <target_id> [relation]"""
    parts = args.split()
    if len(parts) < 2:
        console.print("[red]用法: remove_relation <source_id> <target_id> [relation][/red]")
        console.print("[dim]  不指定 relation 则删除所有 source→target 的边[/dim]")
        return

    source = parts[0]
    target = parts[1]
    relation = parts[2] if len(parts) > 2 else None

    removed = kg.remove_relation(source, target, relation)
    if removed == 0:
        console.print(f"[red]未找到关系: {source} → {target}" +
                       (f" [{relation}]" if relation else "") + "[/red]")
        return

    kg.save_to_file()
    console.print(f"[green]已删除 {removed} 条关系: {source} → {target}" +
                   (f" [{relation}]" if relation else " [全部]") + "[/green]")

    if rag:
        console.print("[dim]正在重建索引...[/dim]")
        rag.rebuild_index()
        console.print("[green]索引重建完成[/green]")


def cmd_list(kg: KnowledgeGraph, entity_type: str = None):
    type_map = {
        "character": "character",
        "人物": "character",
        "faction": "faction",
        "势力": "faction",
        "event": "event",
        "事件": "event",
    }
    et = type_map.get(entity_type, entity_type)
    entities = kg.get_all_entities(et)

    if not entities:
        console.print(f"[red]未找到类型 '{entity_type}' 的实体[/red]")
        return

    table = Table(title=f"实体列表（{entity_type or '全部'}）")
    table.add_column("ID", style="dim")
    table.add_column("名称", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("势力", style="yellow")

    for e in entities:
        table.add_row(
            e["id"],
            e["name"],
            e.get("type", ""),
            e.get("faction", "-"),
        )
    console.print(table)


def cmd_viz(kg: KnowledgeGraph, mode: str = "html"):
    """可视化知识图谱"""
    console.print(f"\n[dim]正在生成知识图谱可视化...[/dim]\n")

    try:
        if mode == "png" or mode == "img":
            path = kg.visualize_matplotlib()
            console.print(f"[green]静态图谱: {path}[/green]")
            console.print("[dim]请用图片查看器打开该文件[/dim]")
        elif mode == "html":
            path = kg.visualize_html()
            console.print(f"[green]交互式图谱: {path}[/green]")
        else:
            console.print(f"[red]未知模式: {mode}，可选: html / png[/red]")
    except ImportError as e:
        console.print(f"[red]缺少依赖: {e}[/red]")
        console.print("[yellow]请安装: pip install matplotlib[/yellow]")
    except Exception as e:
        console.print(f"[red]生成失败: {e}[/red]")


def main():
    print_banner()

    # 初始化
    console.print("[dim]初始化知识图谱...[/dim]")
    kg = KnowledgeGraph("data/three_kingdoms.json")
    show_stats(kg)

    console.print("\n[dim]初始化 RAG 引擎 v4（混合检索+重排序+社区摘要+多跳图+三路融合）...[/dim]")
    try:
        rag = RAGEngine(kg)
    except Exception as e:
        console.print(f"[red]RAG 引擎初始化失败: {e}[/red]")
        console.print("[yellow]将仅支持图查询模式，不支持语义检索[/yellow]")
        rag = None

    console.print("\n[green]初始化完成！输入 [bold]help[/bold] 查看命令，[bold]quit[/bold] 退出[/green]\n")

    # 交互循环
    while True:
        try:
            cmd = console.input("[bold cyan]>>> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]再见！[/yellow]")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if action == "quit" or action == "exit":
            console.print("[yellow]再见！[/yellow]")
            break

        elif action == "help":
            print_help()

        elif action == "stats":
            show_stats(kg)

        elif action == "search":
            if not args:
                console.print("[red]用法: search <名称>[/red]")
            else:
                cmd_search(kg, args)

        elif action == "neighbors":
            if not args:
                console.print("[red]用法: neighbors <实体ID>[/red]")
            else:
                cmd_neighbors(kg, args)

        elif action == "path":
            path_args = args.split()
            if len(path_args) < 2:
                console.print("[red]用法: path <源ID> <目标ID>[/red]")
            else:
                cmd_path(kg, path_args[0], path_args[1])

        elif action == "faction":
            if not args:
                console.print("[red]用法: faction <势力名>[/red]")
            else:
                cmd_faction(kg, args)

        elif action == "list":
            cmd_list(kg, args if args else None)

        elif action == "viz":
            cmd_viz(kg, args if args else "html")

        elif action == "ask":
            if not args:
                console.print("[red]用法: ask <问题>[/red]")
            elif rag is None:
                console.print("[red]RAG 引擎不可用[/red]")
            else:
                cmd_rag(rag, args)

        elif action == "community":
            cmd_community(rag, kg)

        elif action == "extract":
            if not args:
                console.print("[red]用法: extract <文本>[/red]")
            else:
                cmd_extract(args)

        elif action == "eval":
            if rag is None:
                console.print("[red]RAG 引擎不可用，无法评估[/red]")
            else:
                from evaluation import Evaluator
                console.print("[dim]正在运行评估（22 个测试用例，可能需要数十秒）...[/dim]\n")
                evaluator = Evaluator()
                evaluator.evaluate_and_report(rag, verbose=True)

        elif action == "add_entity":
            cmd_add_entity(kg, rag, args)

        elif action == "add_relation":
            cmd_add_relation(kg, rag, args)

        elif action == "remove_entity":
            cmd_remove_entity(kg, rag, args)

        elif action == "remove_relation":
            cmd_remove_relation(kg, rag, args)

        elif action == "serve":
            port = int(args) if args.isdigit() else 8000
            console.print(f"[bold green]启动 Web API 服务，端口 {port}...[/bold green]")
            console.print(f"[dim]API 文档: http://localhost:{port}/docs[/dim]")
            console.print("[dim]按 Ctrl+C 停止[/dim]\n")
            import uvicorn
            from api import app
            uvicorn.run(app, host="0.0.0.0", port=port)

        else:
            console.print(f"[red]未知命令: {action}，输入 help 查看帮助[/red]")


if __name__ == "__main__":
    main()