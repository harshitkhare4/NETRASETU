import os
import sqlite3
import random
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "netrasetu.db")

# ============================================================
# DEMO MODE CONFIGURATION
# ============================================================
# Set DEMO_MODE = True only for explicit platform demonstration.
# In LIVE MODE (default), no synthetic records are seeded.
# Override at runtime via NETRASETU_DEMO_MODE environment variable.
# ============================================================

DEMO_MODE = os.environ.get("NETRASETU_DEMO_MODE", "false").lower() == "true"


def get_db_connection():
    """Returns a SQLite connection with Row factory enabled."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes the NetraSetu screening database schema.
    In LIVE MODE (default): creates schema only — no synthetic data seeded.
    In DEMO MODE: seeds synthetic historical records for platform demonstration.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS screenings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT UNIQUE NOT NULL,
        timestamp TEXT NOT NULL,
        date TEXT NOT NULL,
        month TEXT NOT NULL,
        year INTEGER NOT NULL,
        image_name TEXT NOT NULL,
        source_type TEXT NOT NULL,             -- 'upload', 'camera', 'sample', 'demo'
        quality_status TEXT NOT NULL,          -- 'GOOD', 'REVIEW'
        blur_score REAL,
        brightness REAL,
        contrast REAL,
        retinal_area REAL,
        quality_reasons TEXT,
        predicted_class INTEGER,               -- 0, 1, 2, 3, 4
        icdr_grade TEXT,
        severity TEXT,
        class_confidence REAL,
        referable_score REAL,
        referable INTEGER,                     -- 0 or 1
        processing_status TEXT NOT NULL,       -- 'COMPLETED', 'NEEDS RECAPTURE', 'REQUIRES SPECIALIST REVIEW', 'PROCESSING FAILED'
        review_status TEXT NOT NULL,           -- 'Pending Review', 'Reviewed', 'Needs Further Assessment', 'Not Required'
        review_notes TEXT,
        reviewed_at TEXT,
        processing_time_ms REAL,
        enhanced_path TEXT,
        gradcam_path TEXT,
        report_path TEXT,
        json_path TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS review_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        previous_status TEXT,
        new_status TEXT NOT NULL,
        notes TEXT,
        FOREIGN KEY (case_id) REFERENCES screenings (case_id)
    )
    """)

    conn.commit()
    conn.close()

    # Only seed demo data when DEMO_MODE is explicitly enabled
    if DEMO_MODE:
        seed_demo_data_if_needed()
        print("[NetraSetu DB] Running in DEMO MODE — synthetic screening records loaded.")
    else:
        print("[NetraSetu DB] Running in LIVE MODE — no synthetic records seeded.")


def generate_case_id(date_str=None):
    """Generates an anonymous Case ID in the format NS-YYYY-MM-DD-XXX."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT case_id FROM screenings WHERE date = ?", (date_str,))
    rows = cursor.fetchall()
    max_num = 0
    prefix = f"NS-{date_str}-"
    for r in rows:
        cid = r["case_id"]
        if cid.startswith(prefix):
            try:
                num = int(cid[len(prefix):])
                if num > max_num:
                    max_num = num
            except (ValueError, TypeError):
                pass
    next_num = max(max_num + 1, len(rows) + 1)
    conn.close()

    return f"NS-{date_str}-{next_num:03d}"


def insert_screening(record):
    """Inserts a completed screening record into the database."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    month_str = now.strftime("%Y-%m")
    year = now.year

    conn = get_db_connection()
    cursor = conn.cursor()

    case_id = record.get("case_id") or generate_case_id(date_str)
    if case_id:
        cursor.execute("SELECT 1 FROM screenings WHERE case_id = ?", (case_id,))
        if cursor.fetchone():
            case_id = generate_case_id(date_str)
    timestamp = record.get("timestamp") or now.strftime("%Y-%m-%d %H:%M:%S")

    # Determine processing_status and review_status based on clinical protocol
    quality = record.get("quality", {})
    quality_status = quality.get("status", "GOOD")
    referable = bool(record.get("referable", False))

    processing_status = record.get("processing_status")
    review_status = record.get("review_status")

    if not processing_status:
        if quality_status == "REVIEW":
            processing_status = "NEEDS RECAPTURE"
            review_status = "Not Required"
        elif referable:
            processing_status = "REQUIRES SPECIALIST REVIEW"
            review_status = "Pending Review"
        else:
            processing_status = "COMPLETED"
            review_status = "Not Required"

    if not review_status:
        review_status = "Pending Review" if referable else "Not Required"

    pred_class = record.get("predicted_class")
    if pred_class is None:
        pred_class = -1

    cursor.execute("""
    INSERT INTO screenings (
        case_id, timestamp, date, month, year, image_name, source_type,
        quality_status, blur_score, brightness, contrast, retinal_area, quality_reasons,
        predicted_class, icdr_grade, severity, class_confidence, referable_score,
        referable, processing_status, review_status, review_notes,
        processing_time_ms, enhanced_path, gradcam_path, report_path, json_path
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?, ?, ?, ?
    )
    """, (
        case_id,
        timestamp,
        date_str,
        month_str,
        year,
        record.get("image", "fundus.jpg"),
        record.get("source_type", "upload"),
        quality_status,
        quality.get("blur_score", 0.0),
        quality.get("brightness", 0.0),
        quality.get("contrast", 0.0),
        quality.get("retinal_area_ratio", 0.0),
        ", ".join(quality.get("reasons", [])),
        pred_class,
        record.get("icdr_grade", ""),
        record.get("severity", ""),
        record.get("class_confidence", 0.0),
        record.get("referable_score", 0.0),
        1 if referable else 0,
        processing_status,
        review_status,
        None,
        record.get("processing_time_ms", 120.0),
        record.get("outputs", {}).get("enhanced_image", ""),
        record.get("outputs", {}).get("gradcam") or "",
        record.get("outputs", {}).get("report", ""),
        record.get("outputs", {}).get("json", "")
    ))

    conn.commit()
    conn.close()

    record["case_id"] = case_id
    record["processing_status"] = processing_status
    record["review_status"] = review_status
    return case_id


# ============================================================
# LIVE-ONLY SOURCE FILTER
# Records from live screening activity only: upload and camera.
# 'sample' runs are test/demo samples — excluded from live-mode statistics.
# 'demo' records are synthetic simulated records — always excluded from live stats.
# ============================================================

LIVE_SOURCES = ("'upload'", "'camera'")
LIVE_SOURCES_SQL = ", ".join(LIVE_SOURCES)  # 'upload', 'camera'


def get_history(filter_period=None, search=None, limit=100, offset=0, live_only=None):
    """Retrieves screening history with date filtering and search query.

    Args:
        live_only: If True, show only live/user upload and camera records.
                   If False/None, show all records.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM screenings WHERE 1=1"
    params = []

    # In LIVE MODE, default to showing all non-demo records but label them.
    # If caller explicitly requests live_only=True, filter to upload+camera only.
    if live_only:
        query += f" AND source_type IN ({LIVE_SOURCES_SQL})"

    now = datetime.now()
    if filter_period == "today":
        query += " AND date = ?"
        params.append(now.strftime("%Y-%m-%d"))
    elif filter_period == "week":
        week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        query += " AND date >= ?"
        params.append(week_ago)
    elif filter_period == "month":
        query += " AND month = ?"
        params.append(now.strftime("%Y-%m"))

    if search:
        query += " AND (case_id LIKE ? OR image_name LIKE ? OR icdr_grade LIKE ?)"
        s = f"%{search.strip()}%"
        params.extend([s, s, s])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def _get_live_rows_for_month(cursor, month_str):
    """Returns only live screening activity (upload/camera) records for a given month."""
    cursor.execute(
        f"SELECT * FROM screenings WHERE month = ? AND source_type IN ({LIVE_SOURCES_SQL})",
        (month_str,)
    )
    return [dict(r) for r in cursor.fetchall()]


def _get_all_rows_for_month(cursor, month_str):
    """Returns all records (including demo) for a given month."""
    cursor.execute("SELECT * FROM screenings WHERE month = ?", (month_str,))
    return [dict(r) for r in cursor.fetchall()]


def _compute_monthly_stats(rows, month_str):
    """Computes monthly statistics from a list of row dicts."""
    total = len(rows)
    if total == 0:
        return {
            "month": month_str,
            "total_screenings": 0,
            "completed": 0,
            "needs_recapture": 0,
            "specialist_review": 0,
            "processing_failed": 0,
            "referable": 0,
            "non_referable": 0,
            "dr_distribution": {"Level 0": 0, "Level 1": 0, "Level 2": 0, "Level 3": 0, "Level 4": 0},
            "quality_summary": {"good": 0, "review": 0},
            "avg_confidence": 0.0,
            "avg_processing_time_ms": 0.0,
            "quality_failure_rate": 0.0,
            "referral_rate": 0.0
        }

    completed = sum(1 for r in rows if r["processing_status"] == "COMPLETED")
    needs_recapture = sum(1 for r in rows if r["processing_status"] == "NEEDS RECAPTURE")
    specialist_review = sum(1 for r in rows if r["processing_status"] == "REQUIRES SPECIALIST REVIEW")
    processing_failed = sum(1 for r in rows if r["processing_status"] == "PROCESSING FAILED")

    referable = sum(1 for r in rows if r["referable"] == 1)

    dr_dist = {f"Level {i}": sum(1 for r in rows if r["predicted_class"] == i) for i in range(5)}
    good_quality = sum(1 for r in rows if r["quality_status"] == "GOOD")

    confidences = [r["class_confidence"] for r in rows if r["class_confidence"] is not None and r["class_confidence"] > 0]
    avg_conf = round(sum(confidences) / len(confidences) * 100, 2) if confidences else 0.0

    times = [r["processing_time_ms"] for r in rows if r["processing_time_ms"] is not None]
    avg_time = round(sum(times) / len(times), 1) if times else 0.0

    return {
        "month": month_str,
        "total_screenings": total,
        "completed": completed,
        "needs_recapture": needs_recapture,
        "specialist_review": specialist_review,
        "processing_failed": processing_failed,
        "referable": referable,
        "non_referable": total - referable,
        "dr_distribution": dr_dist,
        "quality_summary": {"good": good_quality, "review": total - good_quality},
        "avg_confidence": avg_conf,
        "avg_processing_time_ms": avg_time,
        "quality_failure_rate": round((needs_recapture / total) * 100, 2) if total > 0 else 0.0,
        "referral_rate": round((referable / total) * 100, 2) if total > 0 else 0.0,
    }


def get_monthly_report(year=None, month=None):
    """Calculates monthly screening statistics.
    In LIVE MODE: only counts live screening activity (upload/camera).
    In DEMO MODE: counts all records including synthetic.
    """
    now = datetime.now()
    if not year:
        year = now.year
    if not month:
        month = now.month

    month_str = f"{year}-{month:02d}"

    conn = get_db_connection()
    cursor = conn.cursor()

    if DEMO_MODE:
        rows = _get_all_rows_for_month(cursor, month_str)
        data_mode = "demo"
        data_mode_label = "DEMO DATA"
        demo_disclosure = (
            "These screening records are simulated for platform demonstration "
            "and represent test/demo activity where applicable."
        )
    else:
        rows = _get_live_rows_for_month(cursor, month_str)
        data_mode = "live"
        data_mode_label = "LIVE DATA"
        demo_disclosure = None

    conn.close()

    stats = _compute_monthly_stats(rows, month_str)
    if total := len(rows):
        stats["month_name"] = datetime.strptime(month_str, "%Y-%m").strftime("%B %Y")
    else:
        stats["month_name"] = datetime.strptime(month_str, "%Y-%m").strftime("%B %Y")

    stats["data_mode"] = data_mode
    stats["data_mode_label"] = data_mode_label
    stats["demo_disclosure"] = demo_disclosure
    return stats


def get_yearly_report(year=None):
    """Calculates yearly summary with month-by-month trends.
    In LIVE MODE: only counts live screening activity (upload/camera).
    In DEMO MODE: counts all records including synthetic.
    """
    if not year:
        year = datetime.now().year

    conn = get_db_connection()
    cursor = conn.cursor()

    if DEMO_MODE:
        cursor.execute("SELECT * FROM screenings WHERE year = ?", (year,))
        rows = [dict(r) for r in cursor.fetchall()]
        data_mode = "demo"
        data_mode_label = "DEMO DATA"
        demo_disclosure = (
            "These screening records are simulated for platform demonstration "
            "and represent test/demo activity where applicable."
        )
    else:
        cursor.execute(
            f"SELECT * FROM screenings WHERE year = ? AND source_type IN ({LIVE_SOURCES_SQL})",
            (year,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        data_mode = "live"
        data_mode_label = "LIVE DATA"
        demo_disclosure = None

    conn.close()

    total = len(rows)
    months_data = []

    for m in range(1, 13):
        m_str = f"{year}-{m:02d}"
        m_rows = [r for r in rows if r["month"] == m_str]
        m_total = len(m_rows)
        m_ref = sum(1 for r in m_rows if r["referable"] == 1)
        m_review = sum(1 for r in m_rows if r["processing_status"] == "REQUIRES SPECIALIST REVIEW")
        m_recapture = sum(1 for r in m_rows if r["processing_status"] == "NEEDS RECAPTURE")

        months_data.append({
            "month_num": m,
            "month_abbr": datetime(year, m, 1).strftime("%b"),
            "total": m_total,
            "referable": m_ref,
            "requires_review": m_review,
            "recapture": m_recapture,
            "referral_rate": round((m_ref / m_total) * 100, 1) if m_total > 0 else 0.0
        })

    completed = sum(1 for r in rows if r["processing_status"] == "COMPLETED")
    needs_recapture = sum(1 for r in rows if r["processing_status"] == "NEEDS RECAPTURE")
    specialist_review = sum(1 for r in rows if r["processing_status"] == "REQUIRES SPECIALIST REVIEW")
    processing_failed = sum(1 for r in rows if r["processing_status"] == "PROCESSING FAILED")
    referable = sum(1 for r in rows if r["referable"] == 1)

    dr_dist = {f"Level {i}": sum(1 for r in rows if r["predicted_class"] == i) for i in range(5)}

    return {
        "year": year,
        "total_screenings": total,
        "completed": completed,
        "needs_recapture": needs_recapture,
        "specialist_review": specialist_review,
        "processing_failed": processing_failed,
        "referable": referable,
        "non_referable": total - referable,
        "overall_referral_rate": round((referable / total) * 100, 2) if total > 0 else 0.0,
        "dr_distribution": dr_dist,
        "monthly_breakdown": months_data,
        "data_mode": data_mode,
        "data_mode_label": data_mode_label,
        "demo_disclosure": demo_disclosure,
    }


def get_review_queue(status=None):
    """Retrieves cases requiring specialist review.
    In LIVE MODE: returns only live screening activity cases (upload/camera). Excludes sample and demo records.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if DEMO_MODE:
        query = "SELECT * FROM screenings WHERE processing_status = 'REQUIRES SPECIALIST REVIEW'"
    else:
        query = (
            f"SELECT * FROM screenings WHERE processing_status = 'REQUIRES SPECIALIST REVIEW'"
            f" AND source_type IN ({LIVE_SOURCES_SQL})"
        )
    params = []

    if status and status != "all":
        query += " AND review_status = ?"
        params.append(status)

    query += " ORDER BY id DESC LIMIT 100"
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def update_review_status(case_id, new_status, notes=None):
    """Updates specialist review status and logs the review timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT review_status FROM screenings WHERE case_id = ?", (case_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Case ID '{case_id}' not found.")

    prev_status = row[0]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        UPDATE screenings
        SET review_status = ?, review_notes = ?
        WHERE case_id = ?
    """, (new_status, notes, case_id))

    cursor.execute("""
        INSERT INTO review_logs (case_id, timestamp, previous_status, new_status, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (case_id, now_str, prev_status, new_status, notes))

    conn.commit()
    conn.close()

    return {
        "case_id": case_id,
        "previous_status": prev_status,
        "new_status": new_status,
        "notes": notes,
        "reviewed_at": now_str
    }


def get_analytics_summary():
    """Returns top-level KPIs for the analytics dashboard.
    In LIVE MODE: total_screenings counts ONLY upload+camera records.
                  sample_runs and demo_records reported separately.
    In DEMO MODE: counts all records.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if DEMO_MODE:
        src_filter = ""
    else:
        src_filter = f" AND source_type IN ({LIVE_SOURCES_SQL})"

    cursor.execute(f"SELECT COUNT(*) FROM screenings WHERE 1=1{src_filter}")
    total = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM screenings WHERE referable = 1{src_filter}")
    referable = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM screenings WHERE processing_status = 'NEEDS RECAPTURE'{src_filter}")
    recaptures = cursor.fetchone()[0]

    cursor.execute(
        f"SELECT COUNT(*) FROM screenings WHERE review_status = 'Pending Review'{src_filter}"
    )
    pending_reviews = cursor.fetchone()[0]

    cursor.execute(f"SELECT AVG(processing_time_ms) FROM screenings WHERE 1=1{src_filter}")
    avg_time = cursor.fetchone()[0] or 0.0

    dr_dist = {}
    for i in range(5):
        cursor.execute(
            f"SELECT COUNT(*) FROM screenings WHERE predicted_class = ?{src_filter}", (i,)
        )
        dr_dist[f"Level {i}"] = cursor.fetchone()[0]

    # Always-present separate counters — NEVER included in LIVE total
    cursor.execute("SELECT COUNT(*) FROM screenings WHERE source_type = 'sample'")
    sample_runs = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM screenings WHERE source_type = 'demo'")
    demo_records = cursor.fetchone()[0]

    conn.close()

    if DEMO_MODE:
        data_mode = "demo"
        data_mode_label = "DEMO DATA"
        demo_disclosure = (
            "These screening records are simulated for platform demonstration "
            "and represent test/demo activity where applicable."
        )
    else:
        data_mode = "live"
        data_mode_label = "LIVE DATA"
        demo_disclosure = None

    return {
        # LIVE total: ONLY upload + camera. Never includes sample or demo.
        "total_screenings": total,
        "referable_cases": referable,
        "non_referable_cases": total - referable,
        "recapture_requests": recaptures,
        "pending_specialist_reviews": pending_reviews,
        "avg_processing_time_ms": round(avg_time, 1),
        "dr_distribution": dr_dist,
        # Separate counters — always present regardless of mode
        "sample_runs": sample_runs,
        "demo_records": demo_records,
        "data_mode": data_mode,
        "data_mode_label": data_mode_label,
        "demo_disclosure": demo_disclosure,
    }


def purge_demo_records():
    """Removes all synthetic demo records (source_type='demo') from the database.
    Safe: never touches records from live screening activity (upload/camera) or test samples.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM screenings WHERE source_type = 'demo'")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"[NetraSetu DB] Purged {deleted} synthetic demo records.")
    return deleted


def purge_seeded_records():
    """
    Removes synthetic seeded records that were inserted by the old auto-seeder.
    Identifies them by the image_name pattern 'fundus_NS-' which is unique to seeded data.
    Safe: does NOT delete records from live screening activity or test samples run via API.
    Returns count of deleted records.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Seeded records always have image_name like 'fundus_NS-2026-01-02-001.jpg'
    cursor.execute("DELETE FROM screenings WHERE image_name LIKE 'fundus_NS-%'")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    print(f"[NetraSetu DB] Purged {deleted} auto-seeded synthetic records.")
    return deleted


def seed_demo_data_if_needed():
    """
    Seeds historical synthetic screening records into the database.
    Only called when DEMO_MODE=True.
    Records are tagged source_type='demo' to distinguish them from live screening activity.
    Uses INSERT OR IGNORE so it is safe to call multiple times.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM screenings WHERE source_type = 'demo'")
    count = cursor.fetchone()[0]

    if count >= 50:
        conn.close()
        return

    print("[NetraSetu DB] Seeding demo screening records (source_type='demo')...")

    icdr_map = {
        0: ("Level 0 - No DR", "No DR"),
        1: ("Level 1 - Mild DR", "Mild DR"),
        2: ("Level 2 - Moderate DR", "Moderate DR"),
        3: ("Level 3 - Severe DR", "Severe DR"),
        4: ("Level 4 - Proliferative DR", "Proliferative DR")
    }

    # Month targets for 2026: realistic rural camp ramp-up
    month_counts = {
        1: 85, 2: 110, 3: 145, 4: 170,
        5: 195, 6: 210, 7: 225, 8: 240, 9: 248
    }

    random.seed(26038)  # Deterministic seed matching Problem Statement ID 26038
    year = 2026

    for month_num, n_cases in month_counts.items():
        month_str = f"{year}-{month_num:02d}"
        days_in_month = 28 if month_num == 2 else 30

        for idx in range(1, n_cases + 1):
            day = min((idx % days_in_month) + 1, days_in_month)
            date_str = f"{year}-{month_num:02d}-{day:02d}"
            case_id = f"DEMO-{date_str}-{idx:03d}"
            hour = random.randint(9, 17)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            timestamp = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"

            is_poor_quality = random.random() < 0.068
            if is_poor_quality:
                quality_status = "REVIEW"
                blur = round(random.uniform(5.0, 14.5), 2)
                brightness = round(random.uniform(15.0, 45.0), 2)
                contrast = round(random.uniform(8.0, 14.0), 2)
                fov = round(random.uniform(0.10, 0.45), 3)
                reasons = "Image appears blurry, Low image contrast"
                pred_class = random.choice([0, 1, 2])
                ref_score = round(random.uniform(0.05, 0.35), 4)
                processing_status = "NEEDS RECAPTURE"
                review_status = "Not Required"
            else:
                quality_status = "GOOD"
                blur = round(random.uniform(85.0, 240.0), 2)
                brightness = round(random.uniform(95.0, 145.0), 2)
                contrast = round(random.uniform(32.0, 55.0), 2)
                fov = round(random.uniform(0.85, 0.99), 3)
                reasons = ""

                weights = [0.42, 0.20, 0.25, 0.09, 0.04]
                pred_class = random.choices([0, 1, 2, 3, 4], weights=weights)[0]

                if pred_class in (2, 3, 4):
                    ref_score = round(random.uniform(0.35, 0.98), 4)
                    processing_status = "REQUIRES SPECIALIST REVIEW"
                    review_status = random.choice(["Pending Review", "Reviewed", "Pending Review"])
                else:
                    ref_score = round(random.uniform(0.01, 0.22), 4)
                    processing_status = "COMPLETED"
                    review_status = "Not Required"

            icdr_name, sev_name = icdr_map[pred_class]
            conf = round(random.uniform(0.52, 0.96), 4)
            proc_time = round(random.uniform(85.0, 145.0), 1)

            cursor.execute("""
            INSERT OR IGNORE INTO screenings (
                case_id, timestamp, date, month, year, image_name, source_type,
                quality_status, blur_score, brightness, contrast, retinal_area, quality_reasons,
                predicted_class, icdr_grade, severity, class_confidence, referable_score,
                referable, processing_status, review_status, review_notes,
                processing_time_ms, enhanced_path, gradcam_path, report_path, json_path
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """, (
                case_id, timestamp, date_str, month_str, year,
                f"demo_{case_id}.jpg", "demo",
                quality_status, blur, brightness, contrast, fov, reasons,
                pred_class, icdr_name, sev_name, conf, ref_score,
                1 if ref_score >= 0.24 else 0,
                processing_status, review_status, None,
                proc_time, "", "", "", ""
            ))

    conn.commit()
    conn.close()
    print("[NetraSetu DB] Demo seeding completed successfully.")


# ============================================================
# Initialize schema on import
# ============================================================
init_db()
