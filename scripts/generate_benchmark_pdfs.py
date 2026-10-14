#!/usr/bin/env python
"""
============================================================================
Project-RAG_LLM — Benchmark PDF & 100-QA Dataset Generator (Phase 2.3)
============================================================================

Programmatically generates:
  1. Three canonical benchmark PDFs:
     - tech_manual.pdf (4 pages, technical architecture, error codes, ports, flags)
     - compliance_policy.pdf (4 pages, clauses, cross-page retention policy, penalties)
     - scanned_operations_log.pdf (3 pages, incident logs, scanned image page for OCR)
  2. Ground-truth 100-QA benchmark dataset (data/eval/benchmark_dataset.json):
     - 40 Semantic questions
     - 30 Lexical / exact identifier questions
     - 15 OCR table / scan questions
     - 15 Unanswerable trap questions (should_abstain = True)
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


def _draw_text_page(doc: fitz.Document, title: str, paragraphs: list[str]) -> fitz.Page:
    """Create a new page and render title and paragraphs cleanly."""
    page = doc.new_page(width=595, height=842)  # A4 standard points
    y = 54.0

    # Draw header line
    page.insert_text((50, y), title, fontsize=14, color=(0.1, 0.2, 0.5))
    y += 24.0
    page.draw_line((50, y), (545, y), color=(0.7, 0.7, 0.8), width=1)
    y += 24.0

    for para in paragraphs:
        rect = fitz.Rect(50, y, 545, min(800.0, y + 160.0))
        # Insert text block with wrapping
        page.insert_textbox(rect, para, fontsize=10.5, fontname="helv", color=(0.15, 0.15, 0.15))
        # Estimate height based on character length
        lines = max(1, len(para) // 65 + para.count("\n") + 1)
        y += lines * 14.0 + 16.0
        if y > 760:
            break

    return page


def build_tech_manual_pdf(output_path: Path) -> Path:
    """Generate 4-page tech_manual.pdf: AetherDB Distributed Cluster Administration Guide v3.4."""
    doc = fitz.open()

    # Page 1: Architecture & Network Topology
    _draw_text_page(
        doc,
        "AetherDB Distributed Cluster Administration Guide v3.4 — Architecture (Page 1)",
        [
            "Section 1.1: Distributed Consensus & Network Topology\n"
            "AetherDB v3.4 is an enterprise-grade distributed key-value and vector database engine. "
            "The cluster operates on a Raft consensus gossip protocol across 3 to 7 nodes per regional zone. "
            "Client RPC applications connect exclusively to Port 9092 for read and write operations. "
            "Inter-node consensus, leader election, and log replication communicate over Port 9093. "
            "Cluster telemetry, health probes, and Prometheus metrics are exposed via HTTP on Port 8080.",
            "Section 1.2: Quorum Mechanics & Fault Tolerance\n"
            "Cluster write consensus requires a strict majority quorum formulated as (N / 2) + 1 active nodes, "
            "where N represents the total configured cluster voting membership. For a 5-node cluster, a quorum of 3 "
            "nodes is mandatory to commit any write transaction to the write-ahead log (WAL). If two nodes fail, "
            "the cluster continues operating seamlessly. If three nodes fail, the cluster enters read-only protective mode.",
            "Section 1.3: Node Roles & State Transitions\n"
            "Nodes transition through three discrete states: Leader, Follower, and Candidate. A heartbeat frequency "
            "maintains leader authority across the peer group. If the leader fails to emit heartbeats, an election timer "
            "triggers candidate state transition and voting rounds begin immediately.",
        ],
    )

    # Page 2: Command-Line Flags & Performance Tuning
    _draw_text_page(
        doc,
        "AetherDB Distributed Cluster Administration Guide v3.4 — Configuration (Page 2)",
        [
            "Section 2.1: Production Command-Line Startup Flags\n"
            "Administrators must tune the following primary flags during production initialization:\n"
            "• --max-connections-per-node=1024 : Establishes the bounded TCP socket connection pool limit per cluster node.\n"
            "• --raft-heartbeat-ms=250 : Dictates the inter-node gossip heartbeat interval in milliseconds (default: 250ms).\n"
            "• --compaction-threshold-mb=512 : Triggers background LSM-tree SSTable compaction when memtable reaches 512 MB.\n"
            "• --snapshot-interval-sec=3600 : Automates periodic point-in-time snapshot serialization every 3600 seconds.\n"
            "• --rx-buffer-mb=128 : Allocates inbound ring buffer space for high-throughput batch vector payloads.",
            "Section 2.2: Memory Allocation Guidelines\n"
            "AetherDB recommends reserving 65% of available system RAM for the unified block cache. "
            "Vector index segments are memory-mapped directly from NVMe storage with asynchronous cache prefetching. "
            "Setting --compaction-threshold-mb=512 prevents disk read amplification while bounding write stalls under 15ms.",
            "Section 2.3: Storage Engine Topology\n"
            "Data is organized hierarchically into write-ahead logs (WAL), in-memory MemTables, and immutable Level-0 to Level-4 SSTables. "
            "Periodic tombstone pruning runs during off-peak compaction cycles to reclaim disk blocks.",
        ],
    )

    # Page 3: Exact Error Codes & Diagnostic Procedures
    _draw_text_page(
        doc,
        "AetherDB Distributed Cluster Administration Guide v3.4 — Diagnostics (Page 3)",
        [
            "Section 3.1: Critical Error Codes and Recovery Playbooks\n"
            "The cluster telemetry subsystem emits standardized alphanumeric error codes when operational anomalies arise:",
            "Error Code ERR_BUFFER_OVERFLOW_0x7A:\n"
            "Indicates that the inbound socket buffer exceeded the 64MB queue capacity during an unexpected traffic surge. "
            "Diagnostic procedure: Verify connection pooling metrics and increase --rx-buffer-mb=128 or scale out client worker nodes.",
            "Error Code ERR_RAFT_SPLIT_BRAIN_0x9F:\n"
            "Signals that multiple leaders were concurrently detected due to an asymmetrical network partition across WAN links. "
            "Diagnostic procedure: Check firewall rules on Port 9093, isolate the minority partition, and restart isolated nodes.",
            "Error Code ERR_TLS_CERT_EXPIRED_0x3C:\n"
            "Occurs when inter-node mutual TLS certificate validation fails due to certificate expiration. "
            "Diagnostic procedure: Renew the internal root CA certificate bundle and execute the aetherdb-cert-reload CLI utility.",
            "Error Code ERR_DISK_IO_STALL_0x11:\n"
            "Fires when WAL sync latency exceeds 500ms for 3 consecutive sync cycles. "
            "Diagnostic procedure: Check underlying NVMe IOPS provisioning and inspect system kernel dmesg for storage bus throttling.",
        ],
    )

    # Page 4: High Availability & Disaster Recovery
    _draw_text_page(
        doc,
        "AetherDB Distributed Cluster Administration Guide v3.4 — Disaster Recovery (Page 4)",
        [
            "Section 4.1: Automated Failover Protocols\n"
            "Automatic leader failover initiates within 750ms when a Raft leader misses 3 consecutive heartbeat intervals (3 * 250ms). "
            "The candidate node with the most up-to-date log index receives affirmative votes from remaining quorum members "
            "and assumes the leader role without dropping in-flight client read requests.",
            "Section 4.2: Recovery Point Objective (RPO) and Recovery Time Objective (RTO)\n"
            "When configured in synchronous replication mode (sync_replication=true), AetherDB guarantees a Recovery Point Objective (RPO) of zero, "
            "meaning zero data loss upon single-node or zone failure. For cross-region disaster recovery replication, "
            "the system delivers a Recovery Time Objective (RTO) under 15 seconds.",
            "Section 4.3: Backup and Restore Verification\n"
            "Daily backup archives are verified using the aetherdb-verify-snapshot integrity scanner. "
            "Encrypted snapshot archives are replicated to offsite object storage and retained for 90 days.",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


def build_compliance_policy_pdf(output_path: Path) -> Path:
    """Generate 4-page compliance_policy.pdf with clauses, cross-page text, and governance rules."""
    doc = fitz.open()

    # Page 1: Encryption & Classification
    _draw_text_page(
        doc,
        "Global Corporate Data Retention and Privacy Governance Framework (SEC-POL-2026) — Page 1",
        [
            "Corporate Policy SEC-POL-2026: Information Security & Data Governance\n"
            "Effective Date: January 1, 2026 | Authority: Global Data Protection Office\n"
            "Scope: All enterprise cloud infrastructure, relational databases, vector storage, and distributed services.",
            "Clause SEC-4.1.2: Mandatory Encryption of Data at Rest\n"
            "All enterprise production data stored in persistent relational databases, vector indexes, and object stores "
            "must be encrypted at rest using AES-256-GCM encryption. Cryptographic envelope keys must be rotated every 90 days "
            "under automated hardware security module (HSM) control. Unencrypted storage of sensitive data is strictly prohibited.",
            "Clause SEC-4.2.1: Data Classification Tiering\n"
            "Enterprise data is partitioned into two regulatory tiers:\n"
            "• TIER-1 RESTRICTED: Personally Identifiable Information (PII), customer payment instruments, medical records, and master credentials.\n"
            "• TIER-2 CONFIDENTIAL: Internal architectural specifications, non-public source code, financial forecasts, and internal communications.",
        ],
    )

    # Page 2: Financial Retention & Start of Cross-Page Text
    _draw_text_page(
        doc,
        "Global Corporate Data Retention and Privacy Governance Framework (SEC-POL-2026) — Page 2",
        [
            "Clause RET-7.3.1: Financial and Accounting Record Retention\n"
            "All general ledger entries, tax filings, audit workpapers, vendor invoices, and financial transaction records "
            "must be retained for a mandatory duration of exactly 7 years (84 months) following the conclusion of the corporate fiscal year. "
            "Premature destruction of financial records constitutes a severe compliance violation subject to immediate regulatory notification.",
            "Clause RET-7.3.4: Telemetry, Logging and Operational Trace Governance (Part A)\n"
            "To balance cybersecurity auditability with data minimization mandates, Clause RET-7.3.4 governs edge telemetry and security audit log destruction, requiring that edge telemetry logs collected from IoT gateways must be retained for exactly 30 days before irreversible cryptographic shredding, whereas",
        ],
    )

    # Page 3: Continuation of Cross-Page Text & DSAR Rules
    _draw_text_page(
        doc,
        "Global Corporate Data Retention and Privacy Governance Framework (SEC-POL-2026) — Page 3",
        [
            "Clause RET-7.3.4: Telemetry, Logging and Operational Trace Governance (Part B — Continued from Page 2)\n"
            "Clause RET-7.3.4 further dictates that security audit traces and privileged access session recordings must be preserved for exactly 180 days to support forensic breach investigations.",
            "Clause DPO-12.4: Data Subject Access Requests (DSAR) and Right to Erasure\n"
            "Data subjects may exercise their right to erasure under international privacy frameworks. "
            "All valid DSAR deletion requests must be fully executed across all production databases, search indexes, "
            "and backup archives within a strict Service Level Agreement of exactly 30 calendar days. "
            "The designated corporate escalation contact is the Chief Privacy Officer via privacy-officer@enterprise-corp.internal.",
            "Clause DPO-12.5: Cross-Border Data Transfer Restrictions\n"
            "Transfer of TIER-1 RESTRICTED customer data across sovereign jurisdictions requires validated Standard Contractual Clauses (SCCs) "
            "and end-to-end transport encryption over TLS 1.3.",
        ],
    )

    # Page 4: Audits & Penalties
    _draw_text_page(
        doc,
        "Global Corporate Data Retention and Privacy Governance Framework (SEC-POL-2026) — Page 4",
        [
            "Clause AUD-9.0.1: Mandatory Annual Compliance Audits\n"
            "The enterprise must undergo formal independent third-party SOC 2 Type II and ISO 27001 compliance examinations "
            "every 12 months. Audit reports must be reviewed by the Board Audit Committee and maintained on file for 5 years.",
            "Clause AUD-9.2.3: Non-Compliance Penalties and Enforcement\n"
            "Violations of this framework are categorized into three severity levels. A Level-3 critical compliance breach "
            "triggers immediate executive committee intervention and mandates formal notification to regulatory supervisory authorities "
            "within 72 hours of incident confirmation. Willful non-compliance may result in disciplinary termination and civil liability.",
            "Clause AUD-9.4.0: Policy Review and Revision Cadence\n"
            "This policy document is subject to mandatory annual re-certification every November by the Chief Information Security Officer.",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


def build_scanned_operations_log_pdf(output_path: Path) -> Path:
    """Generate 3-page scanned_operations_log.pdf with digital incident tables and a rendered scanned image page."""
    doc = fitz.open()

    # Page 1: Native Digital Incident Table
    _draw_text_page(
        doc,
        "Data Center Alpha Operational Shift Incident Logs — Page 1",
        [
            "Data Center Alpha — Operational Incident Shift Log\n"
            "Facility: Ashburn Building 4 | Shift: Night Operations 00:00 - 08:00 UTC",
            "Incident INC-8821:\n"
            "• Incident ID: INC-8821\n"
            "• Timestamp: 2026-03-14 04:12 UTC\n"
            "• Location: Rack Alpha-7 Power Distribution Unit (PDU)\n"
            "• Root Cause: A faulty circuit breaker tripped under unexpected thermal overload during peak batch workloads.\n"
            "• Impact: Redundant PDU secondary feed took over load immediately. Zero server downtime.\n"
            "• Resolution Duration: Resolved in 42 minutes by replacing breaker unit.",
            "Incident INC-9943:\n"
            "• Incident ID: INC-9943\n"
            "• Timestamp: 2026-03-18 11:30 UTC\n"
            "• Location: Top-of-Rack Switch Leaf-04\n"
            "• Root Cause: Top-of-rack network switch firmware crashed due to a memory leak in the internal BGP routing daemon.\n"
            "• Resolution: Switch failover completed automatically; secondary spine rebooted and cleared buffer in 18 minutes.",
        ],
    )

    # Page 2: Pure Scanned / Rendered Image Page (to exercise OCR pipeline)
    # We construct a PIL image of an authentic handwritten/scanned incident log sheet,
    # render it to a JPEG, and insert it as a full-page image with ZERO native selectable text.
    img = Image.new("RGB", (1200, 1600), color=(248, 246, 240))  # Slightly aged paper tint
    draw = ImageDraw.Draw(img)

    # Draw simulated scan border & header lines
    draw.rectangle([60, 60, 1140, 1540], outline=(100, 100, 100), width=3)
    draw.line([60, 160, 1140, 160], fill=(80, 80, 80), width=2)
    draw.line([60, 260, 1140, 260], fill=(120, 120, 120), width=1)

    # Use default bitmap font with large step spacing
    text_lines = [
        ("SCANNED SHIFT LOG SHEET — EMERGENCY INCIDENT REPORT", 100, 100),
        ("FACILITY: DATA CENTER ALPHA — VAULT B", 100, 190),
        ("INCIDENT ID: INC-7712", 100, 300),
        ("TIMESTAMP: 2026-03-22 18:45 UTC", 100, 380),
        ("LOCATION: Rack Delta-12 High Density Compute", 100, 460),
        ("EQUIPMENT TYPE: Liquid Cooled Vector Processing Unit", 100, 540),
        ("ROOT CAUSE: Coolant leak in secondary manifold loop", 100, 620),
        ("DETECTED SENSOR: Moisture Sensor S-441 tripped on tray", 100, 700),
        ("ACTION TAKEN: Emergency valve bypass engaged and auxiliary chillers activated", 100, 780),
        ("CONTAINMENT TIME: Primary loop isolated in 9 minutes", 100, 860),
        ("EQUIPMENT DAMAGE: Zero board damage reported by hardware team", 100, 940),
        ("PRIMARY SHIFT TECHNICIAN: Lead Tech Marcus Vance", 100, 1020),
        ("SUPERVISOR APPROVAL: Approved by Duty Chief Jennifer Ramos", 100, 1100),
        ("STATUS: CLOSED AND ARCHIVED FOR POST-MORTEM", 100, 1180),
    ]

    for line_text, x, y in text_lines:
        draw.text((x, y), line_text, fill=(30, 30, 40))

    img_buffer = BytesIO()
    img.save(img_buffer, format="JPEG", quality=90)
    img_bytes = img_buffer.getvalue()

    # Create page 2 in fitz and insert image covering the whole page
    page2 = doc.new_page(width=595, height=842)
    page2.insert_image(fitz.Rect(0, 0, 595, 842), stream=img_bytes)

    # Page 3: Operational Summary Metrics
    _draw_text_page(
        doc,
        "Data Center Alpha Operational Shift Incident Logs — Summary (Page 3)",
        [
            "Section 3.1: Monthly Operations Metrics and Uptime Performance\n"
            "During the Q1 operational cycle, Data Center Alpha maintained an annualized infrastructure uptime of 99.982% across all 1,024 compute racks. "
            "Total unbudgeted power draw anomalies accounted for less than 1.4 hours of auxiliary diesel generator engagement.",
            "Section 3.2: Cooling System Power Usage Effectiveness (PUE)\n"
            "The liquid cooling infrastructure achieved an average Power Usage Effectiveness (PUE) of 1.14 across high-density GPU racks, "
            "representing an 18% thermal efficiency improvement over legacy forced-air cooling designs.",
            "Section 3.3: Sign-Off and Executive Endorsement\n"
            "All incident remediation records and root cause analyses were reviewed and certified by Operations Director Sarah Jenkins.",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    doc.close()
    return output_path


def generate_100_qa_dataset(dataset_output_path: Path) -> list[dict[str, Any]]:
    """Curate and export exactly 100 benchmark QA items across 4 categories:
    40 semantic, 30 lexical, 15 OCR, and 15 unanswerable trap items.
    """
    items: list[dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SEMANTIC QUESTIONS (40 items, Q001 - Q040)
    # ──────────────────────────────────────────────────────────────────────────
    semantic_data = [
        # AetherDB tech manual semantic items (Q001 - Q020)
        ("How does AetherDB handle node consensus and voting across distributed nodes?",
         "tech_manual.pdf", [1], ["Raft consensus", "majority quorum", "(N / 2) + 1"],
         ["AetherDB uses the Raft consensus protocol across 3 to 7 nodes with a majority quorum requirement of (N / 2) + 1 nodes."]),
        ("What is the quorum formula required to commit write transactions in AetherDB?",
         "tech_manual.pdf", [1], ["majority quorum", "(N / 2) + 1", "write-ahead log"],
         ["Quorum is formulated as (N / 2) + 1 active nodes, meaning a 5-node cluster requires 3 nodes to commit writes."]),
        ("What happens when multiple nodes fail in a five-node AetherDB cluster?",
         "tech_manual.pdf", [1], ["read-only protective mode", "two nodes fail"],
         ["If two nodes fail in a 5-node cluster it continues operating, but if three nodes fail it enters read-only protective mode."]),
        ("How do nodes transition between states when an AetherDB leader fails?",
         "tech_manual.pdf", [1], ["Leader, Follower, and Candidate", "election timer"],
         ["Nodes transition from Follower to Candidate when heartbeats cease, triggering election voting rounds."]),
        ("How does AetherDB balance memory usage between caching and vector indexes?",
         "tech_manual.pdf", [2], ["unified block cache", "65% of available system RAM"],
         ["AetherDB reserves 65% of system RAM for the block cache and memory-maps vector segments directly from NVMe."]),
        ("What storage hierarchy is used by AetherDB to manage writes and persistence?",
         "tech_manual.pdf", [2], ["write-ahead logs (WAL)", "MemTables", "SSTables"],
         ["Data flows through write-ahead logs, in-memory MemTables, and immutable Level-0 to Level-4 SSTables."]),
        ("How does background compaction prevent disk read degradation in AetherDB?",
         "tech_manual.pdf", [2], ["compaction-threshold-mb", "read amplification"],
         ["Background compaction merges SSTables when thresholds are met to prevent read amplification and bound write stalls."]),
        ("What mechanism ensures disaster recovery failover without losing client reads?",
         "tech_manual.pdf", [4], ["Automated leader failover", "most up-to-date log index"],
         ["Automated failover elects the candidate with the most up-to-date log index without dropping client read requests."]),
        ("What are the recovery guarantees provided by AetherDB during regional outages?",
         "tech_manual.pdf", [4], ["Recovery Point Objective (RPO) of zero", "Recovery Time Objective (RTO) under 15 seconds"],
         ["AetherDB guarantees an RPO of zero in synchronous mode and an RTO under 15 seconds for cross-region disaster recovery."]),
        ("How are daily backup snapshots verified and stored in AetherDB?",
         "tech_manual.pdf", [4], ["aetherdb-verify-snapshot", "retained for 90 days"],
         ["Snapshots are verified with aetherdb-verify-snapshot, replicated to offsite object storage, and retained for 90 days."]),
        ("How does AetherDB handle inbound socket traffic surges without crashing?",
         "tech_manual.pdf", [3], ["64MB queue capacity", "rx-buffer-mb=128"],
         ["Traffic surges exceeding 64MB trigger buffer overflow alerts, mitigated by scaling nodes or increasing rx-buffer-mb."]),
        ("Why does network partitioning lead to split-brain states in AetherDB?",
         "tech_manual.pdf", [3], ["asymmetrical network partition", "Port 9093"],
         ["Asymmetrical WAN partitions can isolate nodes such that multiple leaders are elected on Port 9093."]),
        ("What procedure resolves mutual TLS certificate validation failures?",
         "tech_manual.pdf", [3], ["mutual TLS", "aetherdb-cert-reload"],
         ["Administrators must renew the internal root CA certificate bundle and execute aetherdb-cert-reload."]),
        ("What causes storage write stalls during write-ahead log operations?",
         "tech_manual.pdf", [3], ["WAL sync latency exceeds 500ms", "NVMe IOPS"],
         ["Write stalls occur when WAL sync latency exceeds 500ms for 3 consecutive cycles due to NVMe IOPS throttling."]),
        ("What role does the Raft heartbeat interval play in cluster stability?",
         "tech_manual.pdf", [2, 4], ["250ms", "misses 3 consecutive heartbeat intervals"],
         ["The 250ms heartbeat maintains leader authority; missing 3 heartbeats triggers automatic leader failover."]),
        ("How does AetherDB protect against write amplification during heavy ingestion?",
         "tech_manual.pdf", [2], ["compaction-threshold-mb=512", "LSM-tree"],
         ["Setting compaction thresholds to 512MB bounds write stalls under 15ms and controls LSM-tree amplification."]),
        ("What is the diagnostic approach when AetherDB detects split-brain leadership?",
         "tech_manual.pdf", [3], ["isolate the minority partition", "restart isolated nodes"],
         ["Check Port 9093 firewall rules, isolate the minority partition, and restart isolated nodes."]),
        ("How quickly does an AetherDB cluster detect and recover from a failed leader?",
         "tech_manual.pdf", [4], ["initiates within 750ms", "3 * 250ms"],
         ["Failover initiates within 750ms after missing 3 consecutive 250ms heartbeats."]),
        ("What protocol is used for internal metric monitoring in AetherDB?",
         "tech_manual.pdf", [1], ["Prometheus metrics", "Port 8080"],
         ["Prometheus metrics and cluster health telemetry are exposed over HTTP on Port 8080."]),
        ("What guarantees zero data loss in distributed AetherDB deployments?",
         "tech_manual.pdf", [4], ["synchronous replication mode", "sync_replication=true"],
         ["Synchronous replication mode (sync_replication=true) ensures an RPO of zero upon single-node or zone failure."]),

        # Compliance policy semantic items (Q021 - Q040)
        ("What encryption standards are mandated for corporate data storage?",
         "compliance_policy.pdf", [1], ["AES-256-GCM encryption", "envelope keys rotated every 90 days"],
         ["All persistent data must use AES-256-GCM encryption with envelope keys rotated every 90 days."]),
        ("How does the corporate framework classify restricted customer data?",
         "compliance_policy.pdf", [1], ["TIER-1 RESTRICTED", "Personally Identifiable Information (PII)"],
         ["Customer PII and payment data are classified as TIER-1 RESTRICTED, requiring the highest protection standards."]),
        ("What is the distinction between Tier-1 and Tier-2 corporate data classification?",
         "compliance_policy.pdf", [1], ["TIER-1 RESTRICTED", "TIER-2 CONFIDENTIAL"],
         ["Tier-1 covers PII and payment records, while Tier-2 covers internal specifications, source code, and forecasts."]),
        ("Why must financial transaction records be preserved for seven years?",
         "compliance_policy.pdf", [2], ["7 years (84 months)", "general ledger entries"],
         ["Clause RET-7.3.1 mandates 7-year retention for ledgers, invoices, and tax filings from the end of the fiscal year."]),
        ("How does the corporate retention policy balance security auditing with data minimization?",
         "compliance_policy.pdf", [2, 3], ["edge telemetry logs", "security audit traces"],
         ["Edge telemetry is shredded after 30 days, while security audit traces are preserved for 180 days for breach forensics."]),
        ("How long are edge telemetry logs kept before permanent destruction?",
         "compliance_policy.pdf", [2], ["retained for exactly 30 days", "cryptographic shredding"],
         ["Edge telemetry logs collected from IoT gateways are retained for exactly 30 days before cryptographic shredding."]),
        ("How long must security audit traces and privileged access logs be preserved?",
         "compliance_policy.pdf", [3], ["preserved for exactly 180 days", "forensic breach investigations"],
         ["Security audit traces and privileged access session recordings must be preserved for 180 days."]),
        ("What SLA governs corporate responses to data erasure and DSAR requests?",
         "compliance_policy.pdf", [3], ["exactly 30 calendar days", "right to erasure"],
         ["Valid DSAR deletion requests must be completed across all production databases within 30 calendar days."]),
        ("Who is the designated executive escalation contact for privacy and DSAR inquiries?",
         "compliance_policy.pdf", [3], ["Chief Privacy Officer", "privacy-officer@enterprise-corp.internal"],
         ["The Chief Privacy Officer via privacy-officer@enterprise-corp.internal is the designated escalation contact."]),
        ("What requirements govern international cross-border transfers of restricted data?",
         "compliance_policy.pdf", [3], ["Standard Contractual Clauses (SCCs)", "TLS 1.3"],
         ["Cross-border transfers require validated Standard Contractual Clauses (SCCs) and end-to-end TLS 1.3 encryption."]),
        ("What third-party audit certifications are required annually for the enterprise?",
         "compliance_policy.pdf", [4], ["SOC 2 Type II and ISO 27001", "every 12 months"],
         ["The enterprise must undergo independent SOC 2 Type II and ISO 27001 audits every 12 months."]),
        ("What consequences occur if a Level-3 compliance violation is confirmed?",
         "compliance_policy.pdf", [4], ["Level-3 critical compliance breach", "within 72 hours"],
         ["Level-3 violations trigger executive committee intervention and regulatory notification within 72 hours."]),
        ("How often must the corporate privacy and security governance policy be re-certified?",
         "compliance_policy.pdf", [4], ["mandatory annual re-certification every November", "Chief Information Security Officer"],
         ["The policy requires mandatory annual re-certification every November by the Chief Information Security Officer."]),
        ("What storage media are prohibited from storing unencrypted sensitive records?",
         "compliance_policy.pdf", [1], ["relational databases, vector indexes, and object stores", "strictly prohibited"],
         ["Unencrypted storage of sensitive data is prohibited across relational databases, vector indexes, and object stores."]),
        ("What cryptographic key management controls are enforced for data at rest?",
         "compliance_policy.pdf", [1], ["hardware security module (HSM)", "rotated every 90 days"],
         ["Cryptographic envelope keys are rotated every 90 days under automated Hardware Security Module (HSM) control."]),
        ("How are audit records maintained after annual SOC 2 compliance examinations?",
         "compliance_policy.pdf", [4], ["maintained on file for 5 years", "Board Audit Committee"],
         ["SOC 2 Type II audit reports must be reviewed by the Board Audit Committee and maintained on file for 5 years."]),
        ("What disciplinary actions apply to willful violations of corporate data privacy rules?",
         "compliance_policy.pdf", [4], ["disciplinary termination", "civil liability"],
         ["Willful non-compliance may result in disciplinary termination and civil legal liability."]),
        ("Which business documents fall under the 84-month corporate retention rule?",
         "compliance_policy.pdf", [2], ["general ledger entries, tax filings", "7 years (84 months)"],
         ["General ledger entries, tax filings, audit workpapers, and vendor invoices must be retained for 84 months."]),
        ("How does the enterprise ensure customer deletion across search and vector systems?",
         "compliance_policy.pdf", [3], ["production databases, search indexes", "30 calendar days"],
         ["Erasure requests must be executed across databases, search indexes, and backup archives within 30 days."]),
        ("Who oversees the annual compliance re-certification of SEC-POL-2026?",
         "compliance_policy.pdf", [4], ["Chief Information Security Officer", "every November"],
         ["The Chief Information Security Officer oversees the annual re-certification every November."]),
    ]

    for idx, (q, doc_name, pages, evidence, answers) in enumerate(semantic_data, start=1):
        items.append({
            "id": f"Q{idx:03d}",
            "question": q,
            "category": "semantic",
            "expected_doc": doc_name,
            "expected_pages": pages,
            "expected_evidence": evidence,
            "ground_truth_answers": answers,
            "should_abstain": False,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 2. LEXICAL & EXACT CODE QUESTIONS (30 items, Q041 - Q070)
    # ──────────────────────────────────────────────────────────────────────────
    lexical_data = [
        # Tech manual exact codes & parameters (Q041 - Q055)
        ("What port is used for AetherDB client RPC connections?",
         "tech_manual.pdf", [1], ["Port 9092"],
         ["Client RPC applications connect exclusively to Port 9092."]),
        ("What port does AetherDB use for Raft consensus and gossip?",
         "tech_manual.pdf", [1], ["Port 9093"],
         ["Inter-node consensus, leader election, and log replication communicate over Port 9093."]),
        ("Which port exposes Prometheus metrics in AetherDB?",
         "tech_manual.pdf", [1], ["Port 8080"],
         ["Cluster telemetry and Prometheus metrics are exposed via HTTP on Port 8080."]),
        ("What is the exact value of the connection pool limit flag --max-connections-per-node?",
         "tech_manual.pdf", [2], ["--max-connections-per-node=1024"],
         ["The connection pool limit flag is --max-connections-per-node=1024."]),
        ("What is the default Raft heartbeat frequency in milliseconds according to --raft-heartbeat-ms?",
         "tech_manual.pdf", [2], ["--raft-heartbeat-ms=250"],
         ["The default gossip heartbeat interval is --raft-heartbeat-ms=250 (250ms)."]),
        ("What compaction threshold in megabytes is set by --compaction-threshold-mb?",
         "tech_manual.pdf", [2], ["--compaction-threshold-mb=512"],
         ["The compaction threshold is --compaction-threshold-mb=512 (512 MB)."]),
        ("What snapshot interval in seconds is configured by --snapshot-interval-sec?",
         "tech_manual.pdf", [2], ["--snapshot-interval-sec=3600"],
         ["The snapshot interval is --snapshot-interval-sec=3600 (3600 seconds)."]),
        ("What inbound ring buffer size is set by the flag --rx-buffer-mb?",
         "tech_manual.pdf", [2, 3], ["--rx-buffer-mb=128"],
         ["The inbound ring buffer size is --rx-buffer-mb=128."]),
        ("What is the root cause and diagnostic action for error code ERR_BUFFER_OVERFLOW_0x7A?",
         "tech_manual.pdf", [3], ["ERR_BUFFER_OVERFLOW_0x7A", "64MB queue capacity"],
         ["ERR_BUFFER_OVERFLOW_0x7A indicates socket buffer exceeded 64MB queue capacity; fix by increasing rx-buffer-mb."]),
        ("What does error code ERR_RAFT_SPLIT_BRAIN_0x9F signify in the cluster log?",
         "tech_manual.pdf", [3], ["ERR_RAFT_SPLIT_BRAIN_0x9F", "multiple leaders were concurrently detected"],
         ["ERR_RAFT_SPLIT_BRAIN_0x9F signals multiple leaders detected due to a network partition on Port 9093."]),
        ("Which error code indicates that an inter-node mutual TLS certificate has expired?",
         "tech_manual.pdf", [3], ["ERR_TLS_CERT_EXPIRED_0x3C"],
         ["The error code is ERR_TLS_CERT_EXPIRED_0x3C."]),
        ("What CLI tool is executed to reload certificates after ERR_TLS_CERT_EXPIRED_0x3C?",
         "tech_manual.pdf", [3], ["aetherdb-cert-reload"],
         ["The tool is aetherdb-cert-reload."]),
        ("What threshold triggers error code ERR_DISK_IO_STALL_0x11?",
         "tech_manual.pdf", [3], ["ERR_DISK_IO_STALL_0x11", "WAL sync latency exceeds 500ms"],
         ["ERR_DISK_IO_STALL_0x11 fires when WAL sync latency exceeds 500ms for 3 consecutive sync cycles."]),
        ("What utility scans and verifies backup snapshot archives in AetherDB?",
         "tech_manual.pdf", [4], ["aetherdb-verify-snapshot"],
         ["Backup archives are verified using the aetherdb-verify-snapshot integrity scanner."]),
        ("How many consecutive heartbeat intervals must be missed before leader failover initiates?",
         "tech_manual.pdf", [4], ["3 consecutive heartbeat intervals", "750ms"],
         ["Failover initiates within 750ms after missing 3 consecutive heartbeat intervals (3 * 250ms)."]),

        # Compliance policy exact clauses & terms (Q056 - Q070)
        ("What is the official document identifier for the Corporate Governance Framework policy?",
         "compliance_policy.pdf", [1], ["SEC-POL-2026"],
         ["The policy identifier is SEC-POL-2026."]),
        ("Which clause mandates AES-256-GCM encryption of data at rest?",
         "compliance_policy.pdf", [1], ["Clause SEC-4.1.2"],
         ["Clause SEC-4.1.2 mandates AES-256-GCM encryption at rest."]),
        ("What rotation cadence in days is required for cryptographic envelope keys in Clause SEC-4.1.2?",
         "compliance_policy.pdf", [1], ["rotated every 90 days"],
         ["Envelope keys must be rotated every 90 days under automated HSM control."]),
        ("Which policy clause governs data classification tiering?",
         "compliance_policy.pdf", [1], ["Clause SEC-4.2.1"],
         ["Clause SEC-4.2.1 establishes classification into TIER-1 RESTRICTED and TIER-2 CONFIDENTIAL."]),
        ("What clause mandates 7-year retention for financial accounting records?",
         "compliance_policy.pdf", [2], ["Clause RET-7.3.1"],
         ["Clause RET-7.3.1 mandates retention for exactly 7 years (84 months)."]),
        ("How many months is the 7-year financial record retention period equivalent to in Clause RET-7.3.1?",
         "compliance_policy.pdf", [2], ["84 months"],
         ["The 7-year retention period is equivalent to exactly 84 months."]),
        ("Which clause sets the 30-day retention rule for IoT edge telemetry logs?",
         "compliance_policy.pdf", [2], ["Clause RET-7.3.4", "retained for exactly 30 days"],
         ["Clause RET-7.3.4 governs edge telemetry logs with a 30-day retention mandate."]),
        ("According to Clause RET-7.3.4, how many days must security audit traces be preserved?",
         "compliance_policy.pdf", [3], ["Clause RET-7.3.4", "exactly 180 days"],
         ["Security audit traces must be preserved for exactly 180 days under Clause RET-7.3.4."]),
        ("Which clause governs Data Subject Access Requests (DSAR) and Right to Erasure?",
         "compliance_policy.pdf", [3], ["Clause DPO-12.4"],
         ["Clause DPO-12.4 governs DSAR requests and the right to erasure."]),
        ("What is the exact email address for the Chief Privacy Officer specified in Clause DPO-12.4?",
         "compliance_policy.pdf", [3], ["privacy-officer@enterprise-corp.internal"],
         ["The contact email is privacy-officer@enterprise-corp.internal."]),
        ("Which clause governs annual SOC 2 Type II and ISO 27001 audit examinations?",
         "compliance_policy.pdf", [4], ["Clause AUD-9.0.1"],
         ["Clause AUD-9.0.1 mandates annual SOC 2 Type II and ISO 27001 audits."]),
        ("Under Clause AUD-9.2.3, within how many hours must a Level-3 breach be reported to authorities?",
         "compliance_policy.pdf", [4], ["Clause AUD-9.2.3", "within 72 hours"],
         ["Level-3 compliance breaches must be reported to supervisory authorities within 72 hours."]),
        ("How many years must SOC 2 audit reports be kept on file according to Clause AUD-9.0.1?",
         "compliance_policy.pdf", [4], ["maintained on file for 5 years"],
         ["Audit examination reports must be maintained on file for 5 years."]),
        ("What TLS version is mandated for cross-border restricted data transfer in Clause DPO-12.5?",
         "compliance_policy.pdf", [3], ["Clause DPO-12.5", "TLS 1.3"],
         ["Clause DPO-12.5 mandates end-to-end encryption over TLS 1.3."]),
        ("In which month must policy SEC-POL-2026 undergo annual re-certification under Clause AUD-9.4.0?",
         "compliance_policy.pdf", [4], ["Clause AUD-9.4.0", "every November"],
         ["Annual re-certification occurs every November under Clause AUD-9.4.0."]),
    ]

    for idx, (q, doc_name, pages, evidence, answers) in enumerate(lexical_data, start=41):
        items.append({
            "id": f"Q{idx:03d}",
            "question": q,
            "category": "lexical",
            "expected_doc": doc_name,
            "expected_pages": pages,
            "expected_evidence": evidence,
            "ground_truth_answers": answers,
            "should_abstain": False,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 3. OCR & OPERATIONAL LOG QUESTIONS (15 items, Q071 - Q085)
    # ──────────────────────────────────────────────────────────────────────────
    ocr_data = [
        ("What caused the power incident recorded under Incident INC-8821 in Data Center Alpha?",
         "scanned_operations_log.pdf", [1], ["INC-8821", "faulty circuit breaker tripped under unexpected thermal overload"],
         ["A faulty circuit breaker tripped under unexpected thermal overload in Rack Alpha-7 PDU."]),
        ("At what timestamp did incident INC-8821 take place?",
         "scanned_operations_log.pdf", [1], ["INC-8821", "2026-03-14 04:12 UTC"],
         ["Incident INC-8821 occurred on 2026-03-14 at 04:12 UTC."]),
        ("How long did it take to resolve incident INC-8821 in Rack Alpha-7?",
         "scanned_operations_log.pdf", [1], ["Resolved in 42 minutes"],
         ["Incident INC-8821 was resolved in 42 minutes by replacing the breaker unit."]),
        ("What network equipment failed during incident INC-9943?",
         "scanned_operations_log.pdf", [1], ["Top-of-Rack Switch Leaf-04", "INC-9943"],
         ["Top-of-rack network switch Leaf-04 failed during incident INC-9943."]),
        ("What was the root cause of the network switch failure in INC-9943?",
         "scanned_operations_log.pdf", [1], ["memory leak in the internal BGP routing daemon"],
         ["The root cause was a memory leak in the internal BGP routing daemon."]),
        ("How long did switch failover and buffer clearing take during INC-9943?",
         "scanned_operations_log.pdf", [1], ["cleared buffer in 18 minutes"],
         ["Switch failover completed automatically and secondary spine cleared in 18 minutes."]),
        ("What was the incident identifier for the emergency report on the scanned shift log sheet?",
         "scanned_operations_log.pdf", [2], ["INC-7712"],
         ["The scanned emergency incident report was INC-7712."]),
        ("At what timestamp did incident INC-7712 occur on the scanned report?",
         "scanned_operations_log.pdf", [2], ["2026-03-22 18:45 UTC"],
         ["Incident INC-7712 occurred on 2026-03-22 at 18:45 UTC."]),
        ("What rack location was affected by the incident recorded in INC-7712?",
         "scanned_operations_log.pdf", [2], ["Rack Delta-12 High Density Compute"],
         ["The incident affected Rack Delta-12 High Density Compute."]),
        ("What root cause was noted on the scanned log sheet for INC-7712?",
         "scanned_operations_log.pdf", [2], ["Coolant leak in secondary manifold loop"],
         ["The root cause was a coolant leak in the secondary manifold loop."]),
        ("What emergency action was taken to contain the coolant leak in INC-7712?",
         "scanned_operations_log.pdf", [2], ["Emergency valve bypass engaged and auxiliary chillers activated"],
         ["Emergency valve bypass was engaged and auxiliary chillers were activated."]),
        ("Who was the primary shift technician signed off on incident report INC-7712?",
         "scanned_operations_log.pdf", [2], ["Lead Tech Marcus Vance"],
         ["The primary shift technician was Lead Tech Marcus Vance."]),
        ("What annualized infrastructure uptime percentage did Data Center Alpha achieve in Q1?",
         "scanned_operations_log.pdf", [3], ["uptime of 99.982% across all 1,024 compute racks"],
         ["Data Center Alpha maintained an annualized infrastructure uptime of 99.982% across 1,024 compute racks."]),
        ("What Power Usage Effectiveness (PUE) cooling efficiency was recorded for the liquid-cooled GPU racks?",
         "scanned_operations_log.pdf", [3], ["Power Usage Effectiveness (PUE) of 1.14"],
         ["The liquid cooling infrastructure achieved an average PUE of 1.14."]),
        ("Who reviewed and endorsed the Q1 incident remediation records for Data Center Alpha?",
         "scanned_operations_log.pdf", [3], ["Operations Director Sarah Jenkins"],
         ["Operations Director Sarah Jenkins reviewed and certified all remediation records."]),
    ]

    for idx, (q, doc_name, pages, evidence, answers) in enumerate(ocr_data, start=71):
        items.append({
            "id": f"Q{idx:03d}",
            "question": q,
            "category": "ocr",
            "expected_doc": doc_name,
            "expected_pages": pages,
            "expected_evidence": evidence,
            "ground_truth_answers": answers,
            "should_abstain": False,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # 4. UNANSWERABLE TRAP QUESTIONS (15 items, Q086 - Q100) — should_abstain: True
    # ──────────────────────────────────────────────────────────────────────────
    unanswerable_questions = [
        "What are the Friday lunch options on the Data Center Alpha employee cafeteria menu?",
        "What is the AWS root secret key for the company's production S3 backup bucket?",
        "What diagnostic procedure resolves the imaginary quantum error code ERR_QUANTUM_FLUX_0x99?",
        "How many vacation days do junior software engineers receive during their first year?",
        "What is the average orbital velocity of Jupiter's moon Europa in kilometers per second?",
        "Who won the 2024 FIFA World Cup and what was the final match score?",
        "What is the CEO's personal home address and private mobile phone number?",
        "What recipe is recommended for baking traditional sourdough bread at high altitudes?",
        "How do you configure Kubernetes pod autoscaling using KEDA for Apache Kafka consumers?",
        "What is the corporate stock ticker symbol on the Tokyo Stock Exchange?",
        "What is the password to the guest Wi-Fi network in Building 4 Conference Room B?",
        "What is the chemical formula for synthesizing caffeine from methylxanthine precursors?",
        "How many kilometers is it from London Heathrow Airport to Tokyo Haneda Airport?",
        "What error code is triggered when the nonexistent quantum flux capacitor overheats?",
        "What is the corporate dental insurance coverage percentage for root canal treatments?",
    ]

    for idx, q in enumerate(unanswerable_questions, start=86):
        items.append({
            "id": f"Q{idx:03d}",
            "question": q,
            "category": "unanswerable",
            "expected_doc": None,
            "expected_pages": [],
            "expected_evidence": [],
            "ground_truth_answers": [],
            "should_abstain": True,
        })

    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    return items


def main():
    parser = argparse.ArgumentParser(description="Generate 3 benchmark PDFs and 100-QA ground truth dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/benchmark_pdfs"),
        help="Target directory for synthetic PDFs (default: ./data/benchmark_pdfs)",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("./data/eval/benchmark_dataset.json"),
        help="Target JSON path for 100-QA dataset (default: ./data/eval/benchmark_dataset.json)",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    ds_path: Path = args.dataset_output

    out_dir.mkdir(parents=True, exist_ok=True)
    ds_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating canonical benchmark PDFs in:", out_dir)
    tech_pdf = build_tech_manual_pdf(out_dir / "tech_manual.pdf")
    print(f"  [OK] Generated {tech_pdf.name} (pages: {fitz.open(str(tech_pdf)).page_count})")

    comp_pdf = build_compliance_policy_pdf(out_dir / "compliance_policy.pdf")
    print(f"  [OK] Generated {comp_pdf.name} (pages: {fitz.open(str(comp_pdf)).page_count})")

    ocr_pdf = build_scanned_operations_log_pdf(out_dir / "scanned_operations_log.pdf")
    print(f"  [OK] Generated {ocr_pdf.name} (pages: {fitz.open(str(ocr_pdf)).page_count})")

    print(f"\nGenerating 100-QA ground-truth dataset at: {ds_path}")
    items = generate_100_qa_dataset(ds_path)

    cat_counts: dict[str, int] = {}
    abstain_count = 0
    for it in items:
        cat_counts[it["category"]] = cat_counts.get(it["category"], 0) + 1
        if it["should_abstain"]:
            abstain_count += 1

    print(f"  [OK] Exported {len(items)} items to {ds_path}")
    print(f"     Categories: {cat_counts}")
    print(f"     Should-Abstain Trap Items: {abstain_count}")


if __name__ == "__main__":
    main()
