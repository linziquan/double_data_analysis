"""
ReportBuilder — AnalysisPackage 到 Report 的转换层

build_input() 将 AnalysisPackage 转换为报告 AI 的 prompt 输入（向后兼容）。
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional


# ============================================================
# analysis_type → 报告章节名称映射（兼容保留）
# 重要：value 必须与 SECTION_ORDER 中的英文 section_name 完全一致，
# 否则 _format_prompt_input 遍历 SECTION_ORDER 时永远找不到对应 key，
# 导致 AI 收到空 prompt，只输出 3 个固定 section。
# ============================================================
TYPE_TO_SECTION: Dict[str, str] = {
    "growth_analysis": "trend_analysis",
    "ranking_analysis": "ranking_analysis",
    "structure_analysis": "structure_analysis",
    "proportion_analysis": "proportion_analysis",
    "concentration_analysis": "concentration_analysis",
    "distribution_analysis": "distribution_analysis",
    "correlation_analysis": "correlation_analysis",
    "anomaly_analysis": "anomaly_analysis",
    "comparison_analysis": "comparison_analysis",
    "geo_analysis": "geo_analysis",
    "retention_analysis": "retention_analysis",
    # ===== V3 领域模型 analysis_type → 报告章节（2026-08-05 修复：此前缺失导致 V3 包被丢弃、AI 报告为空） =====
    "rfm": "structure_analysis",
    "CLV": "concentration_analysis",
    "cohort": "retention_analysis",
    "churn_rule": "risk_analysis",
    "churn_seg": "risk_analysis",
    "funnel": "funnel_analysis",
    "association_rules": "correlation_analysis",
    "user_profile": "structure_analysis",
    "sku_seg": "structure_analysis",
    "geo_seg": "geo_analysis",
    "activity_seg": "structure_analysis",
    "category_seg": "structure_analysis",
}

SECTION_ORDER = [
    "executive_summary",
    "data_overview",
    "trend_analysis",
    "ranking_analysis",
    "structure_analysis",
    "concentration_analysis",
    "distribution_analysis",
    "correlation_analysis",
    "comparison_analysis",
    "geo_analysis",
    "retention_analysis",
    "funnel_analysis",
    "anomaly_analysis",
    "proportion_analysis",
    "risk_analysis",
    "management_suggestions",
    "action_items",
    "conclusion",
]

# section_name → 中文标题（仅用于人类展示，prompt 解析用英文 key）
SECTION_DISPLAY_NAME: Dict[str, str] = {
    "executive_summary": "执行摘要",
    "data_overview": "数据概览",
    "trend_analysis": "趋势分析",
    "ranking_analysis": "排名分析",
    "structure_analysis": "结构分析",
    "concentration_analysis": "集中度分析",
    "distribution_analysis": "分布分析",
    "correlation_analysis": "相关性分析",
    "comparison_analysis": "对比分析",
    "geo_analysis": "地理空间分析",
    "retention_analysis": "留存分析",
    "funnel_analysis": "转化漏斗分析",
    "anomaly_analysis": "异常分析",
    "proportion_analysis": "占比分析",
    "risk_analysis": "风险分析",
    "management_suggestions": "管理建议",
    "action_items": "下一步行动",
    "conclusion": "总结",
}


class ReportBuilder:
    """将 AnalysisPackage 转换为报告 AI 的 prompt 输入（向后兼容）

    主要方法：
    - build_input() — 为 AI Agent 构建 prompt 输入
    """

    def __init__(self):
        self._section_data: Dict[str, List[Dict[str, Any]]] = {}
        self._data_profile: Dict[str, Any] = {}

    # ===========================================================
    # V2 兼容：build_input() —— AI Agent prompt 构建
    # ===========================================================

    def build_input(
        self,
        packages: List[Dict[str, Any]],
        data_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建 Report AI 的完整输入（兼容保留）

        返回：
        {
            "available_sections": [...],
            "sections_data": {...},
            "data_profile": {...},
            "packages_summary": "...",
            "prompt_text": "..."
        }
        """
        self._section_data = {}
        self._data_profile = data_profile or {}

        for pkg in packages:
            analysis_type = pkg.get("analysis_type", "")
            section_name = TYPE_TO_SECTION.get(analysis_type)
            if section_name is None:
                continue

            extracted = self._extract_package(pkg)
            if extracted is None:
                continue

            if section_name not in self._section_data:
                self._section_data[section_name] = []
            self._section_data[section_name].append(extracted)

        available_sections = list(self._section_data.keys())
        sections_data = self._section_data.copy()

        prompt_text = self._format_prompt_input(sections_data)
        packages_summary = self._build_summary(packages)

        return {
            "available_sections": available_sections,
            "sections_data": sections_data,
            "data_profile": self._data_profile,
            "packages_summary": packages_summary,
            "prompt_text": prompt_text,
        }

    # ===== 内部提取方法（兼容保留） =====

    def _extract_package(self, pkg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从单个 AnalysisPackage 提取关键信息"""
        kpis = pkg.get("kpis", [])
        tables = pkg.get("tables", [])
        insights = pkg.get("insights", [])
        conclusions = pkg.get("conclusions", [])
        recommendations = pkg.get("recommendations", [])
        chart_data = pkg.get("chart_data", [])
        findings = pkg.get("findings", [])
        # 兼容 funnel 等手写模型：图表走 `charts`(ChartItem) 字段而非 `chart_data`。
        # 报告侧统一以 chart_data 为消费入口，故此处把 charts 并入（仅当 chart_data
        # 为空时，避免与其他模型经管线渲染出的 charts 镜像重复）。
        # _safe_chart_data 已对 chart_type/type 做归一化，ChartItem 的 chart_type 可正确读取。
        charts = pkg.get("charts", []) or []
        if not chart_data and charts:
            chart_data = charts

        has_content = bool(kpis or tables or insights or conclusions or chart_data or findings)
        if not has_content:
            return None

        return {
            "analysis_type": pkg.get("analysis_type", ""),
            "business_question": pkg.get("business_question", ""),
            "algorithm": pkg.get("algorithm", ""),
            "dimension": pkg.get("dimension", ""),
            "metric": pkg.get("metric", ""),
            "kpis": self._safe_kpis(kpis),
            "tables": self._safe_tables(tables),
            "insights": self._safe_strings(insights),
            "conclusions": self._safe_strings(conclusions),
            "recommendations": self._safe_strings(recommendations),
            "chart_data": self._safe_chart_data(chart_data),
            "findings": self._safe_findings(findings),
        }

    @staticmethod
    def _safe_kpis(kpis: List[Any]) -> List[Dict[str, Any]]:
        result = []
        for k in kpis:
            if isinstance(k, dict):
                result.append({
                    "label": str(k.get("label", "")),
                    "value": str(k.get("value", "")),
                    "change": str(k.get("change", "")),
                    "type": str(k.get("kpi_type", k.get("type", ""))),
                })
        return result

    @staticmethod
    def _safe_tables(tables: List[Any]) -> List[Dict[str, Any]]:
        result = []
        for t in tables:
            if isinstance(t, dict):
                rows = t.get("rows", [])
                display_rows = rows[:20] if isinstance(rows, list) else []
                result.append({
                    "title": str(t.get("title", "")),
                    "type": str(t.get("table_type", t.get("type", ""))),
                    "columns": [str(c) for c in t.get("columns", [])],
                    "rows": display_rows,
                    "total_rows": len(rows) if isinstance(rows, list) else 0,
                })
        return result

    @staticmethod
    def _safe_chart_data(chart_data: List[Any]) -> List[Dict[str, Any]]:
        result = []
        for c in chart_data:
            if isinstance(c, dict):
                # 保留 option/raw_data/role：V4 报告需要把图表元数据绑定回 section，
                # 若在此处剥离，_bind_package_charts_to_sections 会因 option 为空
                # 跳过所有图，导致报告中「有文字无图表」。
                # 注意：_format_prompt_input 仅输出 title/type/x/y/data_count，
                # option 不会进入 prompt 文本，不浪费 token。
                result.append({
                    "slot": str(c.get("slot", "")),
                    "type": str(c.get("chart_type", c.get("type", ""))),
                    "chart_type": str(c.get("chart_type", c.get("type", ""))),
                    "title": str(c.get("title", "")),
                    "x": str(c.get("x", "")),
                    "y": str(c.get("y", "")),
                    "data_count": len(c.get("data", [])) if isinstance(c.get("data"), list) else 0,
                    # 渲染原料字段：报告链路会用 ChartRenderer 把缺 option 的图现场
                    # 渲染成 ECharts option，data/color/right_col/chart_config 是
                    # ChartData→option 的必需输入。_format_prompt_input 只挑选
                    # title/type/x/y/data_count，这些字段不会进 prompt 文本（token 不变）。
                    "data": c.get("data") if isinstance(c.get("data"), list) else [],
                    "color": str(c.get("color", "")),
                    "right_col": str(c.get("right_col", "")),
                    "chart_config": c.get("chart_config") if isinstance(c.get("chart_config"), dict) else {},
                    "option": c.get("option"),
                    "raw_data": c.get("raw_data"),
                    "role": c.get("role", ""),
                })
        return result

    @staticmethod
    def _safe_findings(findings: List[Any]) -> List[Dict[str, Any]]:
        """保留 BusinessFinding 的序列化 dict（含 business_meaning / business_impact /
        recommendation / severity / evidence），供 _format_prompt_input 透传给 AI。

        这是业务级报告的关键：此前该字段被完全丢弃，AI 只能收到薄的 conclusions。
        category / severity 可能是枚举对象（dataclasses.asdict 不转字符串），
        这里统一归一化为 .value，避免 prompt 中出现 'FindingCategory.GROWTH' 这类脏标签。
        """
        result = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            cat = f.get("category", "")
            sev = f.get("severity", "")
            cat_val = cat.value if hasattr(cat, "value") else str(cat)
            sev_val = sev.value if hasattr(sev, "value") else str(sev)
            result.append({
                "category": cat_val,
                "title": str(f.get("title", "")),
                "description": str(f.get("description", "")),
                "entity": str(f.get("entity", "")),
                "metric": str(f.get("metric", "")),
                "business_meaning": str(f.get("business_meaning", "")),
                "business_impact": str(f.get("business_impact", "")),
                "recommendation": str(f.get("recommendation", "")),
                "severity": sev_val,
                "evidence": f.get("evidence", {}) or {},
            })
        return result

    @staticmethod
    def _safe_strings(items: List[Any]) -> List[str]:
        return [str(i) for i in items if i] if isinstance(items, list) else []

    def _build_summary(self, packages: List[Dict[str, Any]]) -> str:
        lines = [f"共 {len(packages)} 个分析包："]
        for i, pkg in enumerate(packages, 1):
            atype = pkg.get("analysis_type", "未知")
            question = pkg.get("business_question", "")
            # 英文 section_name 转为中文展示名
            section_name = TYPE_TO_SECTION.get(atype, atype)
            display = SECTION_DISPLAY_NAME.get(section_name, section_name)
            lines.append(f"{i}. [{display}] {question}")
        return "\n".join(lines)

    def _format_prompt_input(
        self,
        sections_data: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        if not sections_data:
            return "（无可用分析数据）"

        blocks = []
        for section_name in SECTION_ORDER:
            if section_name not in sections_data:
                continue

            packages = sections_data[section_name]
            # 用中文标题展示（AI 阅读更友好）
            display_name = SECTION_DISPLAY_NAME.get(section_name, section_name)
            blocks.append(f"## {display_name}（{section_name}）")
            blocks.append("")

            for pi, pkg in enumerate(packages, 1):
                if len(packages) > 1:
                    blocks.append(f"### {display_name} #{pi}")

                question = pkg.get("business_question", "")
                if question:
                    blocks.append(f"**业务问题**：{question}")
                    blocks.append("")

                kpis = pkg.get("kpis", [])
                if kpis:
                    kpi_parts = []
                    for k in kpis:
                        change_str = f"（{k['change']}）" if k.get("change") else ""
                        kpi_parts.append(f"{k['label']} {k['value']}{change_str}")
                    blocks.append("**KPI 指标**：" + "；".join(kpi_parts) + "。")
                    blocks.append("")

                tables = pkg.get("tables", [])
                if tables:
                    for t in tables:
                        cols = "、".join(t.get("columns", []))
                        rows = t.get("rows", [])
                        total = t.get("total_rows") or (len(rows) if isinstance(rows, list) else 0)
                        # 取前 3 行作为关键样例，用自然句式概括（不输出 | 分隔的原始表）
                        sample_desc = ""
                        if isinstance(rows, list) and rows:
                            samples = []
                            for row in rows[:3]:
                                if isinstance(row, list):
                                    samples.append("、".join(str(v) for v in row))
                                else:
                                    samples.append(str(row))
                            sample_desc = " 例如：" + "；".join(samples)
                        blocks.append(f"**表格「{t['title']}」**：共 {total} 行，列包括 {cols}。{sample_desc}")
                    blocks.append("")

                insights = pkg.get("insights", [])
                if insights:
                    blocks.append("**数据洞察**：" + "；".join(str(ins).strip() for ins in insights) + "。")
                    blocks.append("")

                conclusions = pkg.get("conclusions", [])
                if conclusions:
                    blocks.append("**分析结论**：" + "；".join(str(c).strip() for c in conclusions) + "。")
                    blocks.append("")

                recommendations = pkg.get("recommendations", [])
                if recommendations:
                    blocks.append("**建议**：" + "；".join(str(r).strip() for r in recommendations) + "。")
                    blocks.append("")

                findings = pkg.get("findings", [])
                if findings:
                    blocks.append("**业务发现（推理依据，优先级最高）**：")
                    for f in findings:
                        if not isinstance(f, dict):
                            continue
                        fcat = f.get("category", "")
                        ftitle = f.get("title", "")
                        fsev = f.get("severity", "")
                        prefix = f"[{fcat}] " if fcat else ""
                        sev = f"（严重度：{fsev}）" if fsev else ""
                        line = f"关于 {prefix}{ftitle}{sev}："
                        parts = []
                        fmeaning = f.get("business_meaning", "")
                        if fmeaning:
                            parts.append(f"业务含义为{fmeaning}")
                        fimpact = f.get("business_impact", "")
                        if fimpact:
                            parts.append(f"业务影响为{fimpact}")
                        frec = f.get("recommendation", "")
                        if frec:
                            parts.append(f"建议{frec}")
                        if parts:
                            line += "；".join(parts) + "。"
                        blocks.append(line)
                    blocks.append("")

                charts = pkg.get("chart_data", [])
                if charts:
                    blocks.append("**关联图表**：")
                    for c in charts:
                        blocks.append(f"- {c['title']}（{c['type']}，X={c['x']}，Y={c['y']}，共 {c['data_count']} 条数据）")
                    blocks.append("")

            blocks.append("---")
            blocks.append("")

        return "\n".join(blocks)


def build_report_input_from_packages(
    packages: List[Dict[str, Any]],
    data_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """快捷函数：一次性构建 Report AI 的完整输入（兼容保留）"""
    builder = ReportBuilder()
    return builder.build_input(packages, data_profile)