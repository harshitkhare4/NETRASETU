import os
import sys
import json
import csv
import io
import logging
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    abort,
    Response
)
from werkzeug.utils import secure_filename

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference_service import get_inference_service, OUTPUT_DIR
import src.db as db
from src.db import DEMO_MODE

# ============================================================
# FLASK CONFIGURATION
# ============================================================

# Ensure runtime directories exist on startup (Render / Docker / local)
os.makedirs(os.path.join(PROJECT_ROOT, "uploads"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results", "inference"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "netrasetu-sih2026-production-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NetraSetuApp")

# Curated valid sample demo catalogue
SAMPLES = {
    "moderate-dr": {
        "id": "moderate-dr",
        "title": "Moderate DR Demo",
        "subtitle": "Level 2 • Processed APTOS",
        "path": os.path.join(PROJECT_ROOT, "data", "APTOS", "processed", "test", "2", "ff52392372d3.jpg"),
        "is_processed": True,
        "description": "Standard processed fundus image exhibiting microaneurysms and hemorrhages."
    },
    "normal-retina": {
        "id": "normal-retina",
        "title": "Normal Retina Demo",
        "subtitle": "Level 0 • No Retinopathy",
        "path": os.path.join(PROJECT_ROOT, "data", "APTOS", "processed", "test", "0", "005b95c28852.jpg"),
        "is_processed": True,
        "description": "Clear fundus with healthy macula, distinct optic disc, and no detectable vascular lesions."
    },
    "proliferative-dr": {
        "id": "proliferative-dr",
        "title": "Proliferative DR Demo",
        "subtitle": "Level 4 • High Risk",
        "path": os.path.join(PROJECT_ROOT, "data", "APTOS", "processed", "test", "4", "02dda30d3acf.jpg"),
        "is_processed": True,
        "description": "Advanced retinopathy with extensive vascular abnormal proliferation and exudates."
    },
    "raw-fundus": {
        "id": "raw-fundus",
        "title": "Raw Fundus Demo",
        "subtitle": "Raw Image • Full Pipeline",
        "path": os.path.join(PROJECT_ROOT, "data", "APTOS", "raw", "train_images", "000c1434d8d7.png"),
        "is_processed": False,
        "description": "Original raw fundus image requiring border cropping, bilateral filtering, and CLAHE illumination enhancement."
    }
}


def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_configured_model_path():
    """Resolves configured model path relative to project root."""
    model_env = os.getenv("NETRASETU_MODEL_PATH", "")
    if model_env:
        if not os.path.isabs(model_env):
            return os.path.normpath(os.path.join(PROJECT_ROOT, model_env))
        return model_env
    return os.path.join(PROJECT_ROOT, "models", "NetraSetu_ResNet50_best.pth")


def check_or_retrieve_model():
    """Checks model presence; attempts automated download if NETRASETU_MODEL_URL is configured."""
    model_path = get_configured_model_path()
    if os.path.exists(model_path):
        return True, model_path

    try:
        from src.download_model import ensure_model_available
        if ensure_model_available(model_path):
            return True, model_path
    except Exception as e:
        logger.error(f"Automated model retrieval error: {e}")

    return False, model_path


# Safe startup logging (no secrets, tokens, or credentials)
_cfg_model_path = get_configured_model_path()
_model_file_exists = os.path.exists(_cfg_model_path)
_research_mode_active = os.getenv("NETRASETU_RESEARCH_MODE", "false").lower() in ("true", "1", "yes")

logger.info("=" * 60)
logger.info("NETRASETU STARTUP INITIALIZATION")
logger.info(f"  Python Version : {sys.version.split()[0]}")
logger.info(f"  PORT           : {os.getenv('PORT', '5000')}")
logger.info(f"  Model Path     : {_cfg_model_path}")
logger.info(f"  Model Exists   : {'YES' if _model_file_exists else 'NO'}")
logger.info(f"  Research Mode  : {'ENABLED' if _research_mode_active else 'DISABLED (Production)'}")
logger.info(f"  Database       : {db.DB_PATH}")
logger.info("=" * 60)

# If model weights are absent but NETRASETU_MODEL_URL is configured, initiate non-blocking retrieval thread
if not _model_file_exists and os.getenv("NETRASETU_MODEL_URL"):
    import threading
    from src.download_model import ensure_model_available
    _bg_dl_thread = threading.Thread(
        target=ensure_model_available,
        args=(_cfg_model_path,),
        name="NetraSetuModelDownloader",
        daemon=True
    )
    _bg_dl_thread.start()
    logger.info(f"[NetraSetu Startup] Initiated background model retrieval for {_cfg_model_path}")


# ============================================================
# ROUTE HANDLERS
# ============================================================

@app.route("/")
def index():
    """Renders the comprehensive NetraSetu Healthcare AI Platform."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    Lightweight health endpoint for Render and external HTTP monitors.
    Returns immediately without loading models, running inference, or performing heavy operations.
    """
    research_mode = os.getenv("NETRASETU_RESEARCH_MODE", "false").lower() in ("true", "1", "yes")
    model_path = get_configured_model_path()
    model_name = os.path.basename(model_path)
    model_available = os.path.exists(model_path)

    if model_available:
        return jsonify({
            "status": "ok",
            "service": "NetraSetu",
            "mode": "research" if research_mode else "production",
            "model": model_name,
            "referable_threshold": 0.24
        }), 200
    else:
        return jsonify({
            "status": "degraded",
            "service": "NetraSetu",
            "mode": "research" if research_mode else "production",
            "model": model_name,
            "model_available": False,
            "message": "Production classifier model is unavailable"
        }), 503


@app.route("/api/health", methods=["GET"])
def api_health():
    """Returns system status, device telemetry, and model readiness."""
    try:
        svc = get_inference_service()
        health_data = svc.get_system_health()
        health_data["database_status"] = "connected"
        return jsonify(health_data), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "error",
            "error": "Failed to retrieve system health",
            "details": str(e)
        }), 500


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Returns metadata for curated demo samples."""
    samples_list = [
        {
            "id": sample["id"],
            "title": sample["title"],
            "subtitle": sample["subtitle"],
            "description": sample["description"],
            "exists": os.path.exists(sample["path"])
        }
        for sample in SAMPLES.values()
    ]
    return jsonify(samples_list), 200


@app.route("/api/sample/<sample_id>", methods=["POST"])
def analyze_sample(sample_id):
    """Executes screening pipeline on a curated demo sample."""
    sample_id = sample_id.lower().strip()
    if sample_id not in SAMPLES:
        return jsonify({"error": f"Unknown sample ID: {sample_id}"}), 404

    model_ready, model_path = check_or_retrieve_model()
    if not model_ready:
        return jsonify({
            "error": "Model unavailable",
            "status": "model_unavailable",
            "model": os.path.basename(model_path),
            "message": "Production classifier model is unavailable on this server. Screening cannot be performed without verified model weights."
        }), 503

    sample_meta = SAMPLES[sample_id]
    sample_path = sample_meta["path"]

    if not os.path.exists(sample_path):
        return jsonify({"error": f"Sample file not found on disk: {sample_meta['title']}"}), 404

    req_json = request.get_json(silent=True) or {}
    override_quality = (request.args.get("override_quality", "false").lower() == "true") or bool(req_json.get("override_quality", False))

    try:
        svc = get_inference_service()
        result = svc.analyze(
            image_input=sample_path,
            filename=os.path.basename(sample_path),
            is_already_processed=sample_meta["is_processed"],
            source_type="sample",
            override_quality=override_quality
        )
        result["sample_info"] = {
            "id": sample_meta["id"],
            "title": sample_meta["title"],
            "description": sample_meta["description"]
        }
        return jsonify(result), 200
    except RuntimeError as re:
        if "classifier weights not available" in str(re) or "classifier is not loaded" in str(re):
            return jsonify({"error": "Model unavailable", "status": "model_unavailable", "message": str(re)}), 503
        return jsonify({"error": "Runtime error", "message": str(re)}), 500
    except Exception as e:
        logger.error(f"Error analyzing sample '{sample_id}': {e}", exc_info=True)
        return jsonify({"error": "Failed to process sample image", "message": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Accepts user-uploaded fundus image (JPG/PNG) and runs inference."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in 'file' parameter"}), 400

    file = request.files["file"]
    if not file or file.filename.strip() == "":
        return jsonify({"error": "No file selected for upload"}), 400

    filename = secure_filename(file.filename)
    if not is_allowed_file(filename):
        return jsonify({"error": "Unsupported file format. Please upload JPG, JPEG, or PNG."}), 400

    model_ready, model_path = check_or_retrieve_model()
    if not model_ready:
        return jsonify({
            "error": "Model unavailable",
            "status": "model_unavailable",
            "model": os.path.basename(model_path),
            "message": "Production classifier model is unavailable on this server. Screening cannot be performed without verified model weights."
        }), 503

    override_quality = (request.form.get("override_quality", "false").lower() == "true") or (request.args.get("override_quality", "false").lower() == "true")

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        svc = get_inference_service()
        result = svc.analyze(
            image_input=image_bytes,
            filename=filename,
            is_already_processed=False,
            source_type="upload",
            override_quality=override_quality
        )
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        if "classifier weights not available" in str(re) or "classifier is not loaded" in str(re):
            return jsonify({"error": "Model unavailable", "status": "model_unavailable", "message": str(re)}), 503
        return jsonify({"error": "Runtime error", "message": str(re)}), 500
    except Exception as e:
        logger.error(f"Inference error in /api/analyze: {e}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred during retinal screening.",
            "message": str(e)
        }), 500


@app.route("/api/analyze-camera", methods=["POST"])
def analyze_camera():
    """Accepts a base64 frame captured from browser webcam and runs inference."""
    data = request.get_json(silent=True) or {}
    data_url = data.get("image_data", "").strip()
    override_quality = bool(data.get("override_quality", False))

    if not data_url:
        return jsonify({"error": "Missing 'image_data' base64 string in request payload."}), 400

    model_ready, model_path = check_or_retrieve_model()
    if not model_ready:
        return jsonify({
            "error": "Model unavailable",
            "status": "model_unavailable",
            "model": os.path.basename(model_path),
            "message": "Production classifier model is unavailable on this server. Screening cannot be performed without verified model weights."
        }), 503

    try:
        svc = get_inference_service()
        result = svc.analyze_camera_frame(data_url, override_quality=override_quality)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        if "classifier weights not available" in str(re) or "classifier is not loaded" in str(re):
            return jsonify({"error": "Model unavailable", "status": "model_unavailable", "message": str(re)}), 503
        return jsonify({"error": "Runtime error", "message": str(re)}), 500
    except Exception as e:
        logger.error(f"Error in /api/analyze-camera: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to analyze webcam captured frame.",
            "message": str(e)
        }), 500


@app.route("/api/experimental-lesions", methods=["POST"])
def experimental_lesions():
    """On-demand execution of IDRiD UNet V2 lesion segmentation."""
    data = request.get_json(silent=True) or {}
    base_name = data.get("base_name", "").strip()

    if not base_name:
        return jsonify({"error": "Parameter 'base_name' is required."}), 400

    base_name = secure_filename(base_name)

    try:
        svc = get_inference_service()
        lesion_result = svc.run_experimental_lesion_analysis(base_name)
        return jsonify(lesion_result), 200
    except FileNotFoundError as fe:
        return jsonify({"error": str(fe)}), 404
    except Exception as e:
        logger.error(f"Lesion segmentation error for '{base_name}': {e}", exc_info=True)
        return jsonify({
            "error": "Failed to run experimental lesion segmentation.",
            "message": str(e)
        }), 500


# ============================================================
# MODE MANAGEMENT ENDPOINTS
# ============================================================

@app.route("/api/mode", methods=["GET"])
def get_mode():
    """Returns the current operating mode (LIVE or DEMO)."""
    return jsonify({
        "mode": "demo" if DEMO_MODE else "live",
        "label": "DEMO MODE" if DEMO_MODE else "LIVE MODE",
        "description": (
            "Simulated screening records are loaded for platform demonstration."
            if DEMO_MODE else
            "Live mode: only live screening activity (upload/camera) is counted."
        )
    }), 200


@app.route("/api/reset-demo-data", methods=["POST"])
def reset_demo_data():
    """
    Removes all synthetic seeded records (source_type='demo') and the old
    auto-seeded records (image_name LIKE 'fundus_NS-%') that were inserted by
    a previous version of the application before the LIVE/DEMO separation.
    Safe: never deletes live screening activity (upload/camera) or test samples.
    """
    try:
        deleted_demo = db.purge_demo_records()
        deleted_seeded = db.purge_seeded_records()
        total_deleted = deleted_demo + deleted_seeded
        return jsonify({
            "message": f"Successfully removed {total_deleted} synthetic records.",
            "deleted_demo": deleted_demo,
            "deleted_seeded": deleted_seeded,
            "total_deleted": total_deleted
        }), 200
    except Exception as e:
        logger.error(f"Failed to reset demo data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================
# REPORTS & HISTORY ENDPOINTS
# ============================================================

@app.route("/api/history", methods=["GET"])
def get_history():
    """Fetches screening history records with period filter and search query."""
    period = request.args.get("filter", "all")
    search = request.args.get("search", "")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    # In LIVE MODE, show all non-demo records but include source info for frontend labelling.
    # Pass live_only=False so sample runs are visible (they're not synthetic seeded records).
    records = db.get_history(filter_period=period, search=search, limit=limit, offset=offset)
    return jsonify({
        "count": len(records),
        "filter": period,
        "search": search,
        "data_mode": "demo" if DEMO_MODE else "live",
        "screenings": records
    }), 200


@app.route("/api/reports/monthly", methods=["GET"])
def monthly_report():
    """Returns monthly screening report metrics."""
    year = request.args.get("year", type=int) or datetime.now().year
    month = request.args.get("month", type=int) or datetime.now().month

    report_data = db.get_monthly_report(year, month)
    return jsonify(report_data), 200


@app.route("/api/reports/yearly", methods=["GET"])
def yearly_report():
    """Returns annual report with month-by-month trends."""
    year = request.args.get("year", type=int) or datetime.now().year
    yearly_data = db.get_yearly_report(year)
    return jsonify(yearly_data), 200


# ============================================================
# SPECIALIST REVIEW QUEUE
# ============================================================

@app.route("/api/review-queue", methods=["GET"])
def review_queue():
    """Returns list of cases flagged as 'REQUIRES SPECIALIST REVIEW'."""
    status = request.args.get("status", "all")
    queue_records = db.get_review_queue(status=status)
    return jsonify({
        "count": len(queue_records),
        "status_filter": status,
        "cases": queue_records
    }), 200


@app.route("/api/review/<case_id>", methods=["POST"])
def update_review(case_id):
    """Updates specialist review status and clinical notes for a flagged case."""
    case_id = secure_filename(case_id)
    data = request.get_json(silent=True) or {}
    new_status = data.get("status", "Reviewed")
    notes = data.get("notes", "")

    valid_statuses = {"Pending Review", "Reviewed", "Needs Further Assessment"}
    if new_status not in valid_statuses:
        return jsonify({"error": f"Invalid status. Must be one of: {valid_statuses}"}), 400

    try:
        updated = db.update_review_status(case_id, new_status, notes)
        return jsonify({
            "message": "Review status updated successfully.",
            "record": updated
        }), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        logger.error(f"Failed to update review for {case_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================
# ANALYTICS & CAPACITY PLANNING
# ============================================================

@app.route("/api/analytics", methods=["GET"])
def analytics():
    """Returns overall platform KPI summaries."""
    summary = db.get_analytics_summary()
    return jsonify(summary), 200


@app.route("/api/capacity", methods=["GET"])
def capacity():
    """
    Simulates annual screening capacity planning for rural deployment (e.g. 100,000 patients/year).
    """
    annual_target = int(request.args.get("target", 100000))
    working_days = int(request.args.get("working_days", 300))
    clinics = int(request.args.get("clinics", 10))
    avg_inference_sec = float(request.args.get("inference_time_sec", 0.12))

    required_per_day = round(annual_target / working_days, 1)
    required_per_clinic_day = round(required_per_day / clinics, 1)

    # 8 operational hours per day = 28,800 seconds per workstation
    ai_daily_capacity_per_workstation = int((8 * 3600) / avg_inference_sec)
    total_ai_capacity_day = ai_daily_capacity_per_workstation * clinics

    # Configurable planning assumption: 32%
    estimated_specialist_reviews_day = round(required_per_day * 0.32, 1)

    return jsonify({
        "parameters": {
            "annual_target": annual_target,
            "working_days": working_days,
            "clinics": clinics,
            "avg_inference_sec": avg_inference_sec,
            "referral_assumption": "Configurable planning assumption: 32%"
        },
        "throughput_metrics": {
            "required_screenings_per_day": required_per_day,
            "required_per_clinic_day": required_per_clinic_day,
            "ai_daily_capacity_per_workstation": ai_daily_capacity_per_workstation,
            "total_ai_capacity_day": total_ai_capacity_day,
            "estimated_specialist_reviews_day": estimated_specialist_reviews_day,
            "capacity_margin_multiplier": round(total_ai_capacity_day / required_per_day, 1)
        },
        "planning_disclosure": (
            "Throughput Capacity Model: Resource & workstation planning simulation based on local GPU inference metrics. "
            "Represents architectural capacity planning for 100,000+ patients/year rural vision programs; not a clinical deployment claim."
        )
    }), 200


@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Exports all screening records into a downloadable CSV file."""
    records = db.get_history(limit=5000)

    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "Case ID", "Timestamp", "Date", "Image Name", "Source Type",
        "Quality Status", "Blur Score", "Brightness", "Contrast",
        "ICDR Grade", "Severity", "Class Confidence", "Referable Score",
        "Referable", "Processing Status", "Review Status", "Processing Time (ms)"
    ]
    writer.writerow(headers)

    for r in records:
        writer.writerow([
            r["case_id"],
            r["timestamp"],
            r["date"],
            r["image_name"],
            r["source_type"],
            r["quality_status"],
            r["blur_score"],
            r["brightness"],
            r["contrast"],
            r["icdr_grade"],
            r["severity"],
            r["class_confidence"],
            r["referable_score"],
            "YES" if r["referable"] == 1 else "NO",
            r["processing_status"],
            r["review_status"],
            r["processing_time_ms"]
        ])

    csv_data = output.getvalue()
    filename = f"netrasetu_screening_history_{datetime.now().strftime('%Y%m%d')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================================
# FILE SERVING & DOWNLOADS
# ============================================================

@app.route("/api/outputs/<path:filename>", methods=["GET"])
def serve_output(filename):
    """Safely serves generated images and artifacts from results/inference/."""
    safe_name = secure_filename(filename)
    target_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(target_path):
        abort(404)
    return send_from_directory(OUTPUT_DIR, safe_name)


@app.route("/api/download/<file_type>/<path:filename>", methods=["GET"])
def download_artifact(file_type, filename):
    """Triggers file download for reports (.txt) and JSON metadata."""
    safe_name = secure_filename(filename)
    target_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(target_path):
        abort(404)

    mimetype = "text/plain" if file_type == "report" else "application/json"
    return send_from_directory(
        OUTPUT_DIR,
        safe_name,
        as_attachment=True,
        download_name=safe_name,
        mimetype=mimetype
    )


# ============================================================
# RESEARCH MODE & 11-MODULE SYSTEM STATUS ENDPOINTS
# ============================================================

RESEARCH_MODE = os.environ.get("NETRASETU_RESEARCH_MODE", "false").lower() in ("true", "1", "yes")
_research_pipeline = None

def get_research_pipeline():
    global _research_pipeline
    if _research_pipeline is None:
        from src.research_pipeline import ResearchInferencePipeline
        _research_pipeline = ResearchInferencePipeline()
    return _research_pipeline


@app.route("/api/system_status", methods=["GET"])
def system_status():
    """Returns comprehensive readiness status across all 11 NetraSetu modules."""
    return jsonify({
        "status": "ready",
        "research_mode_active": RESEARCH_MODE,
        "production_safety_lock": {
            "model_path": "models/NetraSetu_ResNet50_best.pth",
            "referable_threshold": 0.24,
            "status": "LOCKED_AND_PRESERVED"
        },
        "modules": {
            "1_baseline_classifier": {
                "name": "ResNet-50 5-Class Classifier",
                "checkpoint": "NetraSetu_ResNet50_best.pth",
                "status": "Production Active",
                "aptos_test_accuracy": "82.97%",
                "referral_threshold": 0.24
            },
            "2_quality_gate": {
                "name": "Quality Gate & Optical Blur Audit",
                "status": "Active",
                "action": "Immediate Point-of-Care Recapture if REJECT"
            },
            "3_confidence_calibration": {
                "name": "Confidence Calibration (Temperature Scaling)",
                "status": "Calibrated",
                "optimal_temperature": 0.9215,
                "validation_ece": "0.0352 (ECE reduced from 0.0422)"
            },
            "4_gradcam_attention": {
                "name": "Grad-CAM Attention Map",
                "status": "Active",
                "target_layer": "layer4[-1]"
            },
            "5_landmark_localization": {
                "name": "IDRiD Anatomical Landmark Localization",
                "checkpoint": "NetraSetu_IDRiD_Localization_best.pth",
                "status": "Available",
                "targets": ["Optic Disc", "Fovea Centralis"]
            },
            "6_lesion_segmentation_v3": {
                "name": "IDRiD UNet V3 Lesion Segmentation",
                "checkpoint": "NetraSetu_IDRiD_UNet_V3_best.pth",
                "status": "Available",
                "macro_dice": 0.2965,
                "targets": ["Microaneurysm", "Hemorrhage", "Hard Exudate", "Soft Exudate"]
            },
            "7_vessel_segmentation": {
                "name": "DRIVE Retinal Blood Vessel Segmentation",
                "checkpoint": "NetraSetu_DRIVE_Vessel_UNet_best.pth",
                "status": "Available",
                "dice_score": 0.6780,
                "accuracy": "93.73%"
            },
            "8_multi_modal_evidence_fusion": {
                "name": "Unified Multi-Modal Evidence Fusion",
                "status": "Ready",
                "layers": ["Grad-CAM", "Landmarks", "Lesions", "Vessels", "Combined"]
            },
            "9_explainable_screening_reports": {
                "name": "Human-Readable Explainable Screening Reports",
                "status": "Ready",
                "formats": ["Printable HTML", "1080p Composite Summary PNG", "JSON Record"]
            },
            "10_rural_capacity_simulator": {
                "name": "100k-250k Patient Rural Capacity Simulator",
                "status": "Modeled",
                "pipeline_gpu_latency": "325.2 ms",
                "triage_workload_reduction": "82.3%"
            },
            "11_combined_research_models": {
                "name": "Combined V1 & Quality-Filtered V2 Classifiers",
                "checkpoints": ["NetraSetu_ResNet50_combined_best.pth", "NetraSetu_ResNet50_combined_v2_best.pth"],
                "status": "Research-Only Comparison Available"
            }
        }
    }), 200


@app.route("/api/research/status", methods=["GET"])
def research_status():
    """Returns research mode status and capability manifest."""
    return jsonify({
        "research_mode": RESEARCH_MODE,
        "environment_var": "NETRASETU_RESEARCH_MODE",
        "active": RESEARCH_MODE,
        "message": "Research mode enabled." if RESEARCH_MODE else "Research mode disabled (operating in strict production baseline)."
    }), 200


@app.route("/api/research/analyze", methods=["POST"])
def research_analyze():
    """
    Executes the full research pipeline on uploaded image.
    Available when NETRASETU_RESEARCH_MODE=true or requested via API.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in 'file' parameter"}), 400

    file = request.files["file"]
    if not file or file.filename.strip() == "":
        return jsonify({"error": "No file selected for upload"}), 400

    filename = secure_filename(file.filename)
    if not is_allowed_file(filename):
        return jsonify({"error": "Unsupported file format. Please upload JPG, JPEG, or PNG."}), 400

    temp_dir = os.path.join(PROJECT_ROOT, "results", "final_pipeline", "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, filename)
    file.save(temp_path)

    out_dir = os.path.join(PROJECT_ROOT, "results", "final_pipeline", "screening_output")
    try:
        pipeline = get_research_pipeline()
        res = pipeline.process_single_image(temp_path, output_dir=out_dir)
        return jsonify({
            "status": res.get("status", "SUCCESS"),
            "patient_id": res.get("patient_id"),
            "predicted_grade": res.get("predicted_grade"),
            "predicted_label": res.get("predicted_label"),
            "calibrated_confidence": res.get("calibrated_confidence"),
            "referable_score": res.get("referable_score"),
            "referable_status": res.get("referable_status"),
            "quality": res.get("quality"),
            "landmarks": {
                "optic_disc": res.get("optic_disc_prediction"),
                "fovea": res.get("fovea_prediction")
            },
            "lesions": res.get("lesion_summary"),
            "vessels": res.get("vessel_summary"),
            "reports": {
                "html": os.path.basename(res["report_record"]["html_report"]) if res.get("report_record") else None,
                "composite_png": os.path.basename(res["report_record"]["composite_png"]) if res.get("report_record") and res["report_record"].get("composite_png") else None
            },
            "disclaimer": "AI-assisted research screening evaluation. Not a clinical diagnosis."
        }), 200
    except Exception as e:
        logger.error(f"Research inference failed: {e}", exc_info=True)
        return jsonify({"error": "Research screening failed", "details": str(e)}), 500


@app.route("/api/research/report/<path:filename>", methods=["GET"])
def serve_research_report(filename):
    """Serves generated screening reports (HTML / PNG)."""
    safe_name = secure_filename(filename)
    candidates = [
        os.path.join(PROJECT_ROOT, "results", "final_pipeline", "screening_output"),
        os.path.join(PROJECT_ROOT, "results", "final_pipeline", "reports", "sample_screening_reports"),
        os.path.join(PROJECT_ROOT, "results", "final_pipeline", "qa", "regression_outputs")
    ]
    for d in candidates:
        target = os.path.join(d, safe_name)
        if os.path.exists(target):
            return send_from_directory(d, safe_name)
    abort(404)


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    port = 5000
    host = "127.0.0.1"
    print("\n" + "=" * 60)
    print(" NETRASETU - HEALTHCARE AI SCREENING PLATFORM (SIH 2026)")
    print(f" Access URL: http://{host}:{port}/")
    print("=" * 60 + "\n")
    app.run(host=host, port=port, debug=False, threaded=True)
