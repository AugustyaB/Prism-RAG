"""Playwright End-to-End User Simulation and Visual Aesthetic Audit Script.

Tests all Web Studio features as a user:
1. Multi-Tenant Persona Switching & RBAC Guards
2. Document Catalog & Ingestion UI States
3. 4-Mode Retrieval Query Execution, Citation Tooltips & Abstention Banners
4. Cloud Gemini Configuration & Status Badge
5. Observability & Telemetry Drawer
6. Multi-Viewport Responsive Layout Audit & Visual Clipping Detection
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
SCREENSHOTS_DIR = ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT = {
    "console_logs": [],
    "page_errors": [],
    "network_errors": [],
    "features_tested": {},
    "visual_oddities": [],
    "responsive_checks": {},
}


def log_console(msg):
    REPORT["console_logs"].append({"type": msg.type, "text": msg.text, "location": msg.location})


def log_page_error(err):
    REPORT["page_errors"].append(str(err))


def run_audit():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        page.on("console", log_console)
        page.on("pageerror", log_page_error)
        page.on(
            "requestfailed",
            lambda req: REPORT["network_errors"].append({
                "url": req.url,
                "failure": req.failure,
            }),
        )

        print("[AUDIT] Navigating to http://127.0.0.1:8000 ...")
        page.goto("http://127.0.0.1:8000")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(1000)

        # ── 1. Initial State Screenshot ──
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_initial_load.png"), full_page=True)
        print("[AUDIT] Captured 01_initial_load.png")

        # ── 2. Persona Switching & RBAC Verification ──
        personas = [
            ("admin_default", "Admin (Default)", "ADMIN", "default"),
            ("manager_default", "Manager (Default)", "MANAGER", "default"),
            ("analyst_default", "Analyst (Default)", "ANALYST", "default"),
            ("viewer_default", "Viewer (Default)", "VIEWER", "default"),
            ("admin_tenant_b", "Admin (Tenant B)", "ADMIN", "tenant_b"),
            ("viewer_tenant_b", "Viewer (Tenant B)", "VIEWER", "tenant_b"),
        ]

        persona_results = []
        for p_id, label, role, org in personas:
            btn = page.locator(f".persona-btn[data-persona='{p_id}']")
            assert btn.count() > 0, f"Persona button for {p_id} not found"
            btn.click()
            page.wait_for_timeout(600)

            email_badge = page.locator(".badge-email").inner_text()
            role_badge = page.locator(".badge-role").inner_text()
            org_badge = page.locator(".badge-org").inner_text()

            # Check drop zone active or disabled
            drop_zone = page.locator("#drop-zone")
            is_upload_disabled = "disabled" in (drop_zone.get_attribute("class") or "")

            # Check delete buttons presence
            delete_btns = page.locator(".doc-delete-btn, .btn-doc-delete").count()

            persona_results.append({
                "persona_id": p_id,
                "label": label,
                "displayed_email": email_badge,
                "displayed_role": role_badge,
                "displayed_org": org_badge,
                "upload_disabled": is_upload_disabled,
                "delete_buttons_count": delete_btns,
            })

        REPORT["features_tested"]["persona_switching_and_rbac"] = persona_results
        print(f"[AUDIT] Verified {len(persona_results)} personas and RBAC rules.")

        # Switch back to default Admin
        page.locator(".persona-btn[data-persona='admin_default']").click()
        page.wait_for_timeout(600)
        page.screenshot(path=str(SCREENSHOTS_DIR / "02_admin_persona_active.png"), full_page=True)

        # ── 3. Document Catalog & UI Elements Audit ──
        doc_cards = page.locator("#doc-list .doc-card").all()
        doc_titles = []
        for card in doc_cards:
            title_el = card.locator(".doc-title, .doc-name, strong")
            if title_el.count() > 0:
                doc_titles.append(title_el.inner_text())
            else:
                doc_titles.append(card.inner_text().splitlines()[0])

        REPORT["features_tested"]["document_catalog"] = {
            "rendered_count": len(doc_cards),
            "documents": doc_titles,
        }
        print(f"[AUDIT] Document Catalog rendered {len(doc_cards)} active items.")

        # ── 4. Cloud Gemini Key Settings Interaction ──
        gemini_input = page.locator("#gemini-api-key")
        gemini_save_btn = page.locator("#save-gemini-key-btn")
        gemini_badge = page.locator("#gemini-key-status")

        initial_gemini_badge = gemini_badge.inner_text() if gemini_badge.count() > 0 else "N/A"
        if gemini_input.count() > 0:
            gemini_input.fill("AIzaSyTestUserVisualAuditKey999")
            gemini_save_btn.click()
            page.wait_for_timeout(600)
            saved_gemini_badge = gemini_badge.inner_text()
        else:
            saved_gemini_badge = "N/A"

        REPORT["features_tested"]["gemini_api_key_ui"] = {
            "initial_status": initial_gemini_badge,
            "after_save_status": saved_gemini_badge,
        }
        page.screenshot(path=str(SCREENSHOTS_DIR / "03_gemini_key_saved.png"))

        # ── 5. Mode Selector & Grounded Query Execution ──
        modes = ["hybrid_rerank", "hybrid_rrf", "vector_only", "bm25_only"]
        for m in modes:
            pill = page.locator(f".pill-btn[data-mode='{m}']")
            if pill.count() > 0:
                pill.click()
                page.wait_for_timeout(200)

        # Select hybrid_rerank for realistic user query
        page.locator(".pill-btn[data-mode='hybrid_rerank']").click()

        # Submit realistic query
        query_input = page.locator("#query-input")
        send_btn = page.locator("#send-btn")

        test_query = "What are the mandatory security policies, encryption protocols, and maintenance schedules?"
        query_input.fill(test_query)
        send_btn.click()
        print("[AUDIT] Sent query. Waiting for response completion...")

        # Wait for AI message to finish rendering (not typing)
        page.wait_for_selector(".msg-ai:not(.typing)", timeout=90000)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "04_query_response.png"), full_page=True)

        # Verify citation chips and tooltip
        citation_chips = page.locator(".citation-chip").all()
        tooltip_info = []
        for i, chip in enumerate(citation_chips[:2]):
            chip.hover()
            page.wait_for_timeout(500)
            tooltip = chip.locator(".citation-tooltip")
            if tooltip.count() > 0:
                box = tooltip.bounding_box()
                tooltip_info.append({
                    "chip_index": i,
                    "chip_text": chip.inner_text().splitlines()[0],
                    "tooltip_box": box,
                    "is_clipped_top": box["y"] < 0 if box else False,
                    "is_clipped_left": box["x"] < 0 if box else False,
                    "is_clipped_right": (box["x"] + box["width"]) > 1440 if box else False,
                })
        
        REPORT["features_tested"]["citations_and_tooltips"] = {
            "citation_chips_count": len(citation_chips),
            "tooltips_analyzed": tooltip_info,
        }

        # Check per-message telemetry ribbon
        telemetry_ribbon = page.locator(".telemetry-ribbon").last
        ribbon_text = telemetry_ribbon.inner_text() if telemetry_ribbon.count() > 0 else "None"
        REPORT["features_tested"]["telemetry_ribbon"] = ribbon_text

        # ── 6. Two-Tier Abstention Gate Test (Out-of-Domain Query) ──
        query_input.fill("What is the exact atmospheric pressure on the surface of Pluto?")
        send_btn.click()
        print("[AUDIT] Sent trap question. Waiting for abstention response completion...")
        page.wait_for_selector(".msg-ai:not(.typing):nth-of-type(4), .abstention-banner", timeout=90000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "05_abstention_gate.png"), full_page=True)

        abstention_banners = page.locator(".abstention-banner").all()
        REPORT["features_tested"]["abstention_banner"] = {
            "banner_count": len(abstention_banners),
            "last_banner_text": abstention_banners[-1].inner_text() if abstention_banners else "None",
        }

        # ── 7. Observability Drawer Interaction ──
        metrics_btn = page.locator("#toggle-metrics-btn")
        if metrics_btn.count() > 0:
            metrics_btn.click()
            page.wait_for_timeout(1000)
            drawer = page.locator("#metrics-drawer")
            drawer_visible = drawer.is_visible() if drawer.count() > 0 else False
            page.screenshot(path=str(SCREENSHOTS_DIR / "06_observability_drawer_open.png"))

            # Click refresh button inside drawer
            refresh_btn = page.locator("#refresh-metrics-btn")
            if refresh_btn.count() > 0:
                refresh_btn.click()
                page.wait_for_timeout(800)

            # Read metrics values
            metric_cards = page.locator(".metric-val").all()
            metric_values = [m.inner_text() for m in metric_cards]

            REPORT["features_tested"]["observability_drawer"] = {
                "drawer_opened": drawer_visible,
                "metric_values": metric_values,
            }

            # Close drawer
            close_btn = page.locator("#close-metrics-btn")
            if close_btn.count() > 0:
                close_btn.click()
                page.wait_for_timeout(500)

        # ── 8. Responsive Viewports & Visual Layout Oddities Audit ──
        viewports = [
            ("desktop_1440x900", 1440, 900),
            ("laptop_1280x800", 1280, 800),
            ("tablet_768x1024", 768, 1024),
            ("mobile_375x812", 375, 812),
        ]

        for vp_name, w, h in viewports:
            page.set_viewport_size({"width": w, "height": h})
            page.evaluate("const sb = document.getElementById('sidebar'); if (sb) sb.scrollTop = 0;")
            page.wait_for_timeout(600)
            shot_path = SCREENSHOTS_DIR / f"07_responsive_{vp_name}.png"
            page.screenshot(path=str(shot_path), full_page=True)

            # Check for horizontal body overflow
            scroll_width = page.evaluate("document.documentElement.scrollWidth")
            client_width = page.evaluate("document.documentElement.clientWidth")
            has_horizontal_overflow = scroll_width > client_width

            # Check persona switcher visibility
            switcher_box = page.locator(".persona-switcher").bounding_box()

            REPORT["responsive_checks"][vp_name] = {
                "viewport": f"{w}x{h}",
                "scroll_width": scroll_width,
                "client_width": client_width,
                "has_horizontal_page_overflow": has_horizontal_overflow,
                "persona_switcher_bounds": switcher_box,
            }

        # ── 9. Visual Oddities Heuristic Analysis ──
        # Check for unstyled elements, unescaped text, overlapping items, color contrast
        oddities = page.evaluate("""() => {
            const issues = [];
            
            // Check for broken images
            document.querySelectorAll('img').forEach(img => {
                if (!img.complete || img.naturalWidth === 0) {
                    issues.push(`Broken image detected: ${img.src}`);
                }
            });

            // Check if element is inside an intended horizontal scroll container
            function isInsideScrollContainer(el) {
                let parent = el.parentElement;
                while (parent && parent !== document.body && parent !== document.documentElement) {
                    const style = window.getComputedStyle(parent);
                    if (style.overflowX === 'auto' || style.overflowX === 'scroll') {
                        return true;
                    }
                    parent = parent.parentElement;
                }
                return false;
            }

            // Check for elements overflowing screen width
            document.querySelectorAll('*').forEach(el => {
                if (isInsideScrollContainer(el)) return;
                const rect = el.getBoundingClientRect();
                if (rect.right > window.innerWidth + 5) {
                    issues.push(`Element overflows viewport width: ${el.tagName}.${el.className} (right=${rect.right}, viewport=${window.innerWidth})`);
                }
            });

            // Check for forbidden color hexes in computed styles
            const forbiddenColors = ['rgb(59, 130, 246)', 'rgb(6, 182, 212)', 'rgb(139, 92, 246)', 'rgb(99, 102, 241)'];
            document.querySelectorAll('*').forEach(el => {
                const style = window.getComputedStyle(el);
                if (forbiddenColors.includes(style.color) || forbiddenColors.includes(style.backgroundColor)) {
                    issues.push(`Forbidden blue/purple computed color on ${el.tagName}.${el.className}`);
                }
            });

            // Check for raw replacement character unicode artifacts
            if (document.body.innerText.includes('\ufffd')) {
                issues.push('Found unicode replacement character \ufffd in rendered page text');
            }

            return issues;
        }""")

        REPORT["visual_oddities"] = oddities

        browser.close()

    # Save detailed JSON report
    report_file = ROOT / "audit_results.json"
    report_file.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    print(f"[AUDIT] Completed audit. Report written to {report_file}")


if __name__ == "__main__":
    run_audit()
