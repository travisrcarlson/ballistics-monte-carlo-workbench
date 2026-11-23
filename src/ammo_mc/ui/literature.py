"""Scientific Literature Database, Formula Governance & Module Provenance Workbench."""

from __future__ import annotations

import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ammo_mc.data import get_kb
from ammo_mc.literature import (
    build_literature_audit_summary,
    export_literature_bibtex,
    export_literature_csv,
    get_data_table_provenance,
    get_literature_catalog,
    get_provenance_map,
    get_sources_for_module,
    search_literature,
)
from ammo_mc.ui.common import _render_sidebar_header


def tab_literature() -> None:
    """Render the dedicated Scientific Literature & Model Provenance workbench."""
    kb = get_kb()
    summary = build_literature_audit_summary()
    catalog = get_literature_catalog()
    prov_map = get_provenance_map()
    all_categories = sorted({r["category"] for r in catalog})
    all_types = sorted({r["type"] for r in catalog})

    with st.sidebar:
        _render_sidebar_header(
            "📚 Literature & Sources",
            f"{summary['total_sources']} primary references • {summary['total_formulas']} formulas • {summary['mapped_modules']} mapped modules",
        )
        q_filter = st.text_input(
            "🔎 Search Author, Title, DOI or Formula",
            placeholder="e.g. Singer, IATG, Bailey, 4355, Bailey...",
            key="lit_search_query",
        )
        wb_filter = st.selectbox(
            "Filter by Workbench",
            ["All", "sim", "cuas", "3d", "mcdrag", "physics", "iatg", "firing", "validation"],
            key="lit_wb_filter",
        )
        cat_filter = st.selectbox("Topic Category", ["All"] + all_categories, key="lit_cat_filter")
        type_filter = st.selectbox("Source Type", ["All"] + all_types, key="lit_type_filter")
        c_chk1, c_chk2 = st.columns(2)
        with c_chk1:
            doi_only = st.checkbox("DOI Only", value=False, key="lit_doi_only")
        with c_chk2:
            verified_only = st.checkbox("Verified Only", value=True, key="lit_verified_only")

        filtered_sources = search_literature(
            query=q_filter,
            category=cat_filter,
            source_type=type_filter,
            workbench_slug=wb_filter,
            verified_only=verified_only,
            doi_only=doi_only,
        )
        filtered_ids = [r["source_id"] for r in filtered_sources]

        st.divider()
        st.caption(f"Showing **{len(filtered_sources)} / {summary['total_sources']}** matching references")
        st.download_button(
            "📑 Export Citations (.bib)",
            data=export_literature_bibtex(filtered_ids or None),
            file_name="ammo_mc_references.bib",
            mime="application/x-bibtex",
            width="stretch",
            key="lit_dl_bib",
        )
        st.download_button(
            "📊 Export Literature Table (.csv)",
            data=export_literature_csv(filtered_ids or None),
            file_name="ammo_mc_literature.csv",
            mime="text/csv",
            width="stretch",
            key="lit_dl_csv",
        )
        audit_payload = json.dumps(
            {
                "summary": summary,
                "filtered_sources": filtered_sources,
                "module_provenance_count": len(prov_map),
            },
            indent=2,
        )
        st.download_button(
            "🧾 Download Provenance Audit (.json)",
            data=audit_payload,
            file_name="ammo_mc_provenance_audit.json",
            mime="application/json",
            width="stretch",
            key="lit_dl_json",
        )

    st.header("📚 Scientific Literature, Standards & Model Provenance Database")
    st.markdown(
        "Complete, auditable traceability showing **where every equation, aerodynamic drag curve, explosive constant, "
        "biomechanical criterion, sensor law, C-UAS engagement model, and empirical parameter comes from**."
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Primary Sources", f"{summary['total_sources']}", f"{summary['verified_sources']} verified")
    k2.metric("Peer-Reviewed DOIs", f"{summary['doi_sources']}", "Published academic & RFC DOIs")
    k3.metric("Standards & Manuals", f"{summary['standards_and_manuals']}", "NATO STANAG / UNODA / DoD")
    k4.metric(
        "Registered Formulas",
        f"{summary['total_formulas']}",
        f"{summary['flagged_formula_count']} FLAGGED ({', '.join(summary['flagged_formula_ids'])})",
    )
    k5.metric(
        "Mapped Modules",
        f"{summary['mapped_modules']}",
        f"{summary['linked_source_coverage_pct']:.0f}% source traceability",
    )

    tab_trace, tab_db, tab_formulas, tab_tables = st.tabs(
        [
            "🧭 Where Every Model Comes From (Module Traceability)",
            f"📚 Literature Database ({len(filtered_sources)}/{summary['total_sources']})",
            f"🧮 Registered Formulas ({summary['total_formulas']})",
            "🗄️ Empirical Data Tables & Catalogs",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: MODULE-BY-MODULE TRACEABILITY EXPLORER
    # ------------------------------------------------------------------
    with tab_trace:
        wb_names = list(dict.fromkeys(m["workbench"] for m in prov_map))
        c_sel1, c_sel2 = st.columns([1.0, 1.8])
        with c_sel1:
            sel_wb = st.selectbox("Select Workbench", wb_names, key="lit_trace_wb")
        mod_candidates = [m for m in prov_map if m["workbench"] == sel_wb]
        mod_labels = [m["module_name"] for m in mod_candidates]
        with c_sel2:
            sel_mod_label = st.selectbox("Select Engineering Module / Sub-Model", mod_labels, key="lit_trace_mod")

        chosen = next(m for m in mod_candidates if m["module_name"] == sel_mod_label)
        dossier = get_sources_for_module(chosen["entry_id"])

        with st.container(border=True):
            st.subheader(f"🔬 {dossier['module_name']}")
            st.markdown(dossier["description"])
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.caption("**Implementing Python Modules**")
                st.code("\n".join(dossier["code_files"]), language="text")
            with m_col2:
                st.caption("**Validity Envelope & Engineering Assumptions**")
                st.info(dossier["assumptions_and_limits"])

            if dossier["has_flagged_formulas"]:
                st.warning(
                    "⚠️ This module invokes one or more FLAGGED Knowledge Base equations "
                    "(governance rules restrict flagged models from serving as sole safety-distance basis)."
                )

        c_src_col, c_eq_col = st.columns([1.25, 1.0])
        with c_src_col:
            st.markdown(f"#### 📖 Governing Literature & Standards ({len(dossier['sources'])})")
            for s in dossier["sources"]:
                badge = f"DOI: `{s['doi']}`" if s["doi"] else s["type"]
                with st.expander(f"[{s['source_id']}] {s['title']} ({s['year'] or 'n/a'})", expanded=True):
                    st.markdown(
                        f"**Authors:** {s['authors']}  \n"
                        f"**Type / Identifier:** {s['type']} • {badge}  \n"
                        f"**Role in Workbench:** {s['used_for']}  \n"
                        f"**Status:** `{s['verification_status']}` • [Open Reference Link]({s['url']})"
                    )

        with c_eq_col:
            st.markdown(f"#### 🧮 Governing KB Equations ({len(dossier['formulas'])})")
            if dossier["formulas"]:
                for f in dossier["formulas"]:
                    flag_txt = "🚨 FLAG" if f.get("is_flagged") else "✅ STD/VERIFIED"
                    with st.expander(f"[{f['formula_id']}] {f['name']} ({flag_txt})", expanded=False):
                        st.markdown(f"**Equation:** `{f['equation_math']}`")
                        st.markdown(f"**Code form:** `{f['code_form']}`")
                        st.markdown(f"**Units & Variables:** {f['variables_units']}")
                        st.markdown(f"**Validity:** {f['validity_assumptions']}")
                        st.caption(f"Sources: {f['source_id']} | Status: {f['verification_status']}")
            else:
                st.info(
                    "Uses specialized numerical solvers & empirical algorithms in domain code "
                    "(Kalman trackers, 3D ray-casting, nonlinear least-squares) governed directly by the cited papers on the left."
                )

            st.markdown(f"#### 🗄️ Data Tables Referenced ({len(dossier['resolved_tables'])})")
            if dossier["resolved_tables"]:
                for dt in dossier["resolved_tables"]:
                    st.markdown(f"- **`{dt['table_name']}`** — {dt.get('description', '')}")
            else:
                st.caption("No external JSON tables required; driven cleanly by user scenario inputs & analytic models.")

    # ------------------------------------------------------------------
    # TAB 2: MASTER SCIENTIFIC LITERATURE DATABASE
    # ------------------------------------------------------------------
    with tab_db:
        cat_items = sorted(summary["category_counts"].items(), key=lambda kv: kv[1])
        fig_cat = go.Figure(
            go.Bar(
                x=[v for _, v in cat_items],
                y=[k for k, _ in cat_items],
                orientation="h",
                marker_color="#3B82F6",
                text=[str(v) for _, v in cat_items],
                textposition="outside",
            )
        )
        fig_cat.update_layout(
            title="Scientific Literature Coverage by Engineering Domain",
            xaxis_title="Number of Curated Sources",
            yaxis_title="",
            height=340,
            margin=dict(l=20, r=40, t=45, b=35),
        )
        st.plotly_chart(fig_cat, width="stretch")

        lit_df = pd.DataFrame(
            [
                {
                    "ID": r["source_id"],
                    "Year": r["year"],
                    "Category": r["category"],
                    "Title": r["title"],
                    "Authors / Authority": r["authors"],
                    "Type": r["type"],
                    "DOI": r["doi"] or "—",
                    "Formulas": r["linked_formula_count"],
                    "Modules": r["linked_module_count"],
                    "Status": r["verification_status"],
                }
                for r in filtered_sources
            ]
        )
        st.dataframe(lit_df, width="stretch", height=360)

        st.markdown("#### Detailed Citation Cards & BibTeX Snippets")
        for r in filtered_sources:
            doi_badge = f" • DOI: `{r['doi']}`" if r["doi"] else ""
            with st.expander(f"**{r['source_id']}** — {r['title']} ({r['year'] or 'n/a'}){doi_badge}"):
                c1, c2 = st.columns([1.6, 1.0])
                with c1:
                    st.markdown(
                        f"**Authors / Issuer:** {r['authors']}  \n"
                        f"**Category:** `{r['category']}` • **Type:** {r['type']}  \n"
                        f"**What We Use It For:** {r['used_for']}  \n"
                        f"**Reference URL:** [{r['url']}]({r['url']})"
                    )
                    if r["used_by_modules"]:
                        st.markdown(f"**Used by Modules:** {', '.join(r['used_by_modules'][:8])}")
                    if r["linked_formula_ids"]:
                        st.markdown(f"**Linked KB Formulas:** `{', '.join(r['linked_formula_ids'])}`")
                with c2:
                    st.caption("BibTeX Citation")
                    st.code(export_literature_bibtex([r["source_id"]]), language="bibtex")

    # ------------------------------------------------------------------
    # TAB 3: FORMULA & GOVERNANCE REGISTRY
    # ------------------------------------------------------------------
    with tab_formulas:
        fc1, fc2 = st.columns([1.2, 1.0])
        f_modules = ["All"] + sorted({str(f["module"]) for f in kb.formulas})
        with fc1:
            sel_f_mod = st.selectbox("Filter by Physics Module", f_modules, key="lit_form_mod")
        with fc2:
            only_flag = st.checkbox("Show Only FLAGGED Equations (Governance Review)", value=False, key="lit_flag_only")

        f_rows = [
            f
            for f in kb.formulas
            if (sel_f_mod == "All" or f["module"] == sel_f_mod) and (not only_flag or f.get("is_flagged"))
        ]
        st.caption(f"Displaying **{len(f_rows)} / {len(kb.formulas)}** governed Knowledge Base formulas.")
        f_df = pd.DataFrame(
            [
                {
                    "Formula ID": f["formula_id"],
                    "Module": f["module"],
                    "Name": f["name"],
                    "Equation": f["equation_math"],
                    "Sources": f["source_id"],
                    "Status": f["verification_status"],
                    "Flagged": bool(f.get("is_flagged")),
                }
                for f in f_rows
            ]
        )
        st.dataframe(f_df, width="stretch", height=420)

    # ------------------------------------------------------------------
    # TAB 4: EMPIRICAL DATA TABLES & CATALOGS
    # ------------------------------------------------------------------
    with tab_tables:
        dt_rows = get_data_table_provenance()
        dt_df = pd.DataFrame(
            [
                {
                    "Table / Catalog": r["table_name"],
                    "Domain": r["domain"],
                    "Volume": r["record_summary"],
                    "Governing Sources": ", ".join(r.get("source_ids", [])),
                    "Description": r["description"],
                }
                for r in dt_rows
            ]
        )
        st.dataframe(dt_df, width="stretch", height=380)

        with st.expander("🔍 Inspect Raw Knowledge Base Empirical Samples", expanded=False):
            sample_choice = st.selectbox(
                "Preview Table",
                ["Explosives (TNT Equiv & Gurney)", "THOR 17 Target Materials", "Lanz-Odermatt / De Marre", "Munition Taxonomy (30 Classes)"],
                key="lit_tbl_preview",
            )
            if sample_choice.startswith("Explosives"):
                st.dataframe(pd.DataFrame(kb.explosives.get("tnt_equivalence", [])), width="stretch")
            elif sample_choice.startswith("THOR"):
                st.dataframe(pd.DataFrame(kb.thor.get("materials", [])), width="stretch")
            elif sample_choice.startswith("Lanz"):
                st.dataframe(pd.DataFrame(kb.penetration.get("lanz_odermatt", [])), width="stretch")
            else:
                st.dataframe(pd.DataFrame(kb.taxonomy), width="stretch")
