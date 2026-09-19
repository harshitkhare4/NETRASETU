"""
src/generate_explainable_report.py
Human-Readable Explainable Screening Report Generator for NetraSetu.

Generates:
1. Self-contained, printable HTML screening report (with base64 embedded multi-layer visualizations)
2. High-resolution Composite Visual Summary PNG
3. Structured JSON screening record
"""

import os
import sys
import json
import base64
import datetime
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

sys.path.insert(0, r"C:\NetraSetu")
from src.explainability_engine import ExplainabilityEngine

def image_to_base64(img_rgb):
    """Encodes an RGB numpy array to base64 PNG string."""
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG", quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def generate_html_report(result, patient_id="NS-PAT-2026-0814", image_path=""):
    """
    Generates a modern, printable clinical research screening report HTML document.
    """
    now_str = datetime.datetime.now().strftime("%d %B %Y, %H:%M:%S IST")
    q = result['quality']
    
    # Check if rejected
    is_rejected = (result.get('status') == 'NEEDS RECAPTURE')
    
    # Badges
    q_color = "#10b981" if q['status'] == "GOOD" else ("#f59e0b" if q['status'] == "REVIEW" else "#ef4444")
    
    if is_rejected:
        pred_grade = "N/A (Recapture Required)"
        grade_badge = "#ef4444"
        ref_badge = "#ef4444"
        ref_text = "INCONCLUSIVE — QUALITY GATE REJECT"
        conf_pct = 0.0
        p_ref = 0.0
    else:
        pred_grade = result['predicted_label']
        conf_pct = result['calibrated_confidence'] * 100
        p_ref = result['referable_score'] * 100
        is_referable = ("REFERABLE" in result['referable_status'] and "NON" not in result['referable_status'])
        ref_badge = "#ef4444" if is_referable else "#10b981"
        ref_text = result['referable_status']
        grade_badge = "#ef4444" if ("Severe" in pred_grade or "Proliferative" in pred_grade) else ("#f59e0b" if "Moderate" in pred_grade or "Mild" in pred_grade else "#10b981")

    # Encode images if present
    b64_orig = image_to_base64(result['images']['original']) if 'images' in result and 'original' in result['images'] else ""
    b64_gradcam = image_to_base64(result['images']['gradcam']) if 'images' in result and 'gradcam' in result['images'] else ""
    b64_combined = image_to_base64(result['images']['combined_evidence']) if 'images' in result and 'combined_evidence' in result['images'] else ""

    # Lesion rows
    lesion_html = ""
    if not is_rejected and 'lesion_summary' in result:
        for name, info in result['lesion_summary'].items():
            status_cls = "badge-danger" if info['status'] == "detected" else ("badge-warning" if info['status'] == "low-confidence" else "badge-muted")
            color_dot = "#ef4444" if name == "Hemorrhage" else ("#06b6d4" if name == "Hard Exudate" else ("#d946ef" if name == "Soft Exudate" else "#eab308"))
            lesion_html += f"""
            <tr>
                <td><span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:{color_dot}; margin-right:8px;"></span><strong>{name}</strong></td>
                <td><span class="badge {status_cls}">{info['status'].upper()}</span></td>
                <td>{info['pixel_count']:,} px</td>
                <td>{info['threshold_applied']}</td>
            </tr>
            """

    # Probability bars
    prob_bars_html = ""
    if not is_rejected and 'calibrated_class_probabilities' in result:
        class_labels = ["Grade 0 (No DR)", "Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (Proliferative)"]
        for label, prob in zip(class_labels, result['calibrated_class_probabilities']):
            pct = prob * 100
            bar_color = "#3b82f6" if pct < 20 else ("#10b981" if "No DR" in label else "#f59e0b")
            prob_bars_html += f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px;">
                    <span>{label}</span>
                    <span><strong>{pct:.1f}%</strong></span>
                </div>
                <div style="background:#e2e8f0; height:8px; border-radius:4px; overflow:hidden;">
                    <div style="background:{bar_color}; width:{pct}%; height:100%;"></div>
                </div>
            </div>
            """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>NetraSetu AI Screening Report — {patient_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 30px;
        }}
        .report-container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
            overflow: hidden;
        }}
        .report-header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #ffffff;
            padding: 24px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title h1 {{
            margin: 0 0 4px 0;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .header-title p {{
            margin: 0;
            font-size: 13px;
            color: #94a3b8;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-success {{ background: #dcfce7; color: #15803d; }}
        .badge-warning {{ background: #fef3c7; color: #b45309; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        .badge-muted {{ background: #f1f5f9; color: #64748b; }}
        .content-body {{
            padding: 32px;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
        }}
        .card-header {{
            font-size: 14px;
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 14px;
            border-bottom: 1px solid #f1f5f9;
            padding-bottom: 8px;
        }}
        .meta-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .meta-table td {{
            padding: 6px 0;
            border-bottom: 1px solid #f8fafc;
        }}
        .meta-table td:first-child {{
            color: #64748b;
            width: 45%;
        }}
        .evidence-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
            margin-top: 16px;
        }}
        .evidence-card {{
            text-align: center;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px;
            background: #fafafa;
        }}
        .evidence-card img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
        }}
        .evidence-card p {{
            margin: 8px 0 0 0;
            font-size: 12px;
            font-weight: 600;
            color: #334155;
        }}
        .disclaimer-banner {{
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 4px solid #f59e0b;
            border-radius: 6px;
            padding: 16px;
            margin-top: 24px;
            font-size: 12px;
            color: #92400e;
            line-height: 1.5;
        }}
        .disclaimer-banner strong {{
            color: #78350f;
        }}
        table.data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        table.data-table th {{
            text-align: left;
            background: #f8fafc;
            padding: 8px 10px;
            border-bottom: 2px solid #e2e8f0;
            color: #475569;
        }}
        table.data-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #f1f5f9;
        }}
    </style>
</head>
<body>

<div class="report-container">
    <!-- Header -->
    <div class="report-header">
        <div class="header-title">
            <h1>NetraSetu — Clinical Screening Report</h1>
            <p>Smart India Hackathon 2026 | Problem Statement 26038 | Explainable Retinal AI</p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #94a3b8;">Patient ID / Session</div>
            <div style="font-size: 16px; font-weight: 700; color: #38bdf8;">{patient_id}</div>
        </div>
    </div>

    <div class="content-body">
        <!-- Screening Summary Cards -->
        <div class="grid-3">
            <div class="card" style="border-top: 4px solid {grade_badge};">
                <div class="card-header">DR Severity Grade</div>
                <div style="font-size: 20px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">{pred_grade}</div>
                <div style="font-size: 13px; color: #64748b;">Calibrated Confidence: <strong>{conf_pct:.1f}%</strong></div>
            </div>

            <div class="card" style="border-top: 4px solid {ref_badge};">
                <div class="card-header">Referral Risk (P &ge; 24%)</div>
                <div style="font-size: 18px; font-weight: 700; color: {ref_badge}; margin-bottom: 4px;">{ref_text}</div>
                <div style="font-size: 13px; color: #64748b;">Referable Probability: <strong>{p_ref:.1f}%</strong></div>
            </div>

            <div class="card" style="border-top: 4px solid {q_color};">
                <div class="card-header">Quality Gate Audit</div>
                <div style="font-size: 18px; font-weight: 700; color: {q_color}; margin-bottom: 4px;">{q['status']}</div>
                <div style="font-size: 13px; color: #64748b;">Blur: {q['blur_score']} | Contrast: {q['contrast']}</div>
            </div>
        </div>

        <!-- Details Grid -->
        <div class="grid-2">
            <!-- Left: Calibrated Probabilities & Landmarks -->
            <div class="card">
                <div class="card-header">Calibrated Class Distribution</div>
                {prob_bars_html}

                <div class="card-header" style="margin-top: 24px;">Anatomical Landmarks</div>
                <table class="meta-table">
                    <tr>
                        <td>Optic Disc Center:</td>
                        <td><strong>({result.get('optic_disc_prediction', {}).get('x', 'N/A')}, {result.get('optic_disc_prediction', {}).get('y', 'N/A')})</strong></td>
                    </tr>
                    <tr>
                        <td>Fovea Centralis:</td>
                        <td><strong>({result.get('fovea_prediction', {}).get('x', 'N/A')}, {result.get('fovea_prediction', {}).get('y', 'N/A')})</strong></td>
                    </tr>
                    <tr>
                        <td>Retinal Vessel Density:</td>
                        <td><strong>{result.get('vessel_summary', {}).get('vessel_density', 0.0)*100:.2f}%</strong></td>
                    </tr>
                </table>
            </div>

            <!-- Right: Lesion Summary -->
            <div class="card">
                <div class="card-header">Lesion Segmentation (IDRiD UNet V3)</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Pathology</th>
                            <th>Status</th>
                            <th>Area</th>
                            <th>Thresh</th>
                        </tr>
                    </thead>
                    <tbody>
                        {lesion_html}
                    </tbody>
                </table>

                <div class="card-header" style="margin-top: 24px;">Session Information</div>
                <table class="meta-table">
                    <tr>
                        <td>Generated On:</td>
                        <td>{now_str}</td>
                    </tr>
                    <tr>
                        <td>Source File:</td>
                        <td><span style="font-size: 11px; word-break: break-all;">{os.path.basename(image_path)}</span></td>
                    </tr>
                    <tr>
                        <td>Temperature Scaling:</td>
                        <td>T = 0.9215 (NLL-optimal)</td>
                    </tr>
                </table>
            </div>
        </div>

        <!-- Multi-Modal Visual Evidence Strip -->
        <div class="card">
            <div class="card-header">Multi-Modal Explainability Evidence</div>
            <div class="evidence-grid">
                <div class="evidence-card">
                    <img src="{b64_orig}" alt="Original Fundus">
                    <p>1. Original Fundus</p>
                </div>
                <div class="evidence-card">
                    <img src="{b64_gradcam}" alt="Grad-CAM Attention">
                    <p>2. Grad-CAM Attention Map</p>
                </div>
                <div class="evidence-card">
                    <img src="{b64_combined}" alt="Multi-Modal Evidence Fusion">
                    <p>3. Unified Multi-Modal Fusion</p>
                </div>
            </div>
        </div>

        <!-- Non-Clinical Research Disclaimer -->
        <div class="disclaimer-banner">
            <strong>RESEARCH SCREENING DISCLAIMER & LEGAL NOTICE:</strong><br>
            This automated screening report was generated by <strong>NetraSetu AI</strong> for research evaluation and clinical workflow decision support. 
            <strong>This is an AI-assisted screening assessment and DOES NOT constitute a definitive medical diagnosis.</strong> Clinical validation has not been performed for autonomous diagnostic use. 
            All findings, including grade predictions, detected lesions, and anatomical localization markers, MUST be verified by a certified ophthalmologist or retinal specialist before any medical intervention or treatment planning.
        </div>
    </div>
</div>

</body>
</html>
"""
    return html

def generate_composite_summary_image(result, patient_id="NS-PAT-2026-0814", image_path=""):
    """
    Generates a 1920x1080 high-resolution visual evidence summary card.
    """
    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas[:] = (248, 250, 252) # Clean off-white
    
    # Top Header Banner
    cv2.rectangle(canvas, (0, 0), (1920, 100), (15, 23, 42), -1)
    cv2.putText(canvas, "NetraSetu — Explainable Retinal Screening Summary", (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Smart India Hackathon 2026 | PS 26038 | Multi-Modal Clinical Decision Support", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"PATIENT: {patient_id}", (1550, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (56, 189, 248), 2, cv2.LINE_AA)

    # 4 Visual Panels: (Original, Grad-CAM, Lesions & Landmarks, Unified Fusion)
    h_panel, w_panel = 440, 440
    y_panels = 140
    xs = [40, 510, 980, 1450]
    
    # Resize and place panels
    orig = cv2.resize(result['images']['original'], (w_panel, h_panel))
    gradcam = cv2.resize(result['images']['gradcam'], (w_panel, h_panel))
    
    # Extract combined
    combined = cv2.resize(result['images']['combined_evidence'], (w_panel, h_panel))
    
    # Vessel panel
    vessel_panel = cv2.resize(result['images']['original'], (w_panel, h_panel))
    vessel_mask_resized = cv2.resize(result['images']['vessel_mask'], (w_panel, h_panel))
    vessel_panel[vessel_mask_resized > 0] = [16, 185, 129]
    vessel_panel = cv2.addWeighted(vessel_panel, 0.45, cv2.resize(result['images']['original'], (w_panel, h_panel)), 0.55, 0)

    panels = [
        (orig, "1. Original Fundus Input"),
        (gradcam, "2. ResNet-50 Grad-CAM Attention"),
        (vessel_panel, "3. DRIVE Vessel Network Segmentation"),
        (combined, "4. Multi-Modal Evidence Fusion")
    ]

    for i, (p_img, p_title) in enumerate(panels):
        x = xs[i]
        # Border
        cv2.rectangle(canvas, (x - 2, y_panels - 2), (x + w_panel + 2, y_panels + h_panel + 2), (203, 213, 225), 2)
        canvas[y_panels:y_panels+h_panel, x:x+w_panel] = cv2.cvtColor(p_img, cv2.COLOR_RGB2BGR)
        # Title banner below panel
        cv2.rectangle(canvas, (x - 2, y_panels + h_panel + 2), (x + w_panel + 2, y_panels + h_panel + 35), (30, 41, 59), -1)
        cv2.putText(canvas, p_title, (x + 10, y_panels + h_panel + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Lower Section: Left Dashboard Card (600 to 1000)
    card_y = 630
    cv2.rectangle(canvas, (40, card_y), (850, 990), (255, 255, 255), -1)
    cv2.rectangle(canvas, (40, card_y), (850, 990), (226, 232, 240), 1)

    cv2.putText(canvas, "SCREENING METRICS & CLINICAL TRIAGE", (60, card_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (15, 23, 42), 2, cv2.LINE_AA)
    
    # Grade & Confidence
    grade = result.get('predicted_label', 'N/A')
    conf = result.get('calibrated_confidence', 0.0) * 100
    ref_status = result.get('referable_status', 'N/A')
    p_ref = result.get('referable_score', 0.0) * 100
    
    cv2.putText(canvas, f"Predicted DR Grade: {grade}", (60, card_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 41, 59), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Calibrated Confidence (T=0.9215): {conf:.1f}%", (60, card_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (71, 85, 105), 1, cv2.LINE_AA)
    
    color_ref = (34, 197, 94) if "NON" in ref_status else (239, 68, 68)
    cv2.putText(canvas, f"Referral Status: {ref_status}", (60, card_y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_ref, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Referable Risk Score: {p_ref:.1f}% (Threshold: 24.0%)", (60, card_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (71, 85, 105), 1, cv2.LINE_AA)

    q = result['quality']
    cv2.putText(canvas, f"Image Quality Status: {q['status']} (Blur: {q['blur_score']:.1f}, Contrast: {q['contrast']:.1f})", (60, card_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 118, 110), 1, cv2.LINE_AA)
    
    od = result.get('optic_disc_prediction', {})
    fov = result.get('fovea_prediction', {})
    cv2.putText(canvas, f"Anatomical Landmarks: OD ({od.get('x', 'N/A')}, {od.get('y', 'N/A')}) | Fovea ({fov.get('x', 'N/A')}, {fov.get('y', 'N/A')})", (60, card_y + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (71, 85, 105), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Vessel Network Density: {result.get('vessel_summary', {}).get('vessel_density', 0.0)*100:.2f}%", (60, card_y + 310), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (71, 85, 105), 1, cv2.LINE_AA)

    # Right Card: Lesion Segmentation Breakdown (880 to 1880)
    cv2.rectangle(canvas, (880, card_y), (1880, 990), (255, 255, 255), -1)
    cv2.rectangle(canvas, (880, card_y), (1880, 990), (226, 232, 240), 1)

    cv2.putText(canvas, "LESION PATHOLOGY FINDINGS (IDRiD UNet V3)", (900, card_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (15, 23, 42), 2, cv2.LINE_AA)
    
    lesion_lines = [
        ("Hemorrhages (HE)", result.get('lesion_summary', {}).get('Hemorrhage', {}), (239, 68, 68)),
        ("Hard Exudates (EX)", result.get('lesion_summary', {}).get('Hard Exudate', {}), (6, 182, 212)),
        ("Soft Exudates (SE)", result.get('lesion_summary', {}).get('Soft Exudate', {}), (217, 70, 239)),
        ("Microaneurysms (MA)", result.get('lesion_summary', {}).get('Microaneurysm', {}), (234, 179, 8))
    ]

    ly = card_y + 90
    for name, info, col in lesion_lines:
        status = info.get('status', 'not detected').upper()
        px = info.get('pixel_count', 0)
        cv2.circle(canvas, (915, ly - 5), 8, col, -1)
        cv2.putText(canvas, f"{name}:", (935, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 41, 59), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Status: {status} | Area: {px:,} pixels | Threshold: {info.get('threshold_applied', 0.5)}", (1180, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (71, 85, 105), 1, cv2.LINE_AA)
        ly += 45

    # Bottom Disclaimer Bar
    cv2.rectangle(canvas, (0, 1020), (1920, 1080), (241, 245, 249), -1)
    disclaimer_text = "RESEARCH NOTICE: AI-assisted screening assessment only. Not a clinical diagnosis. Clinical validation not performed. Ophthalmologist review mandatory."
    cv2.putText(canvas, disclaimer_text, (280, 1055), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 83, 9), 1, cv2.LINE_AA)

    return canvas

def generate_report_for_image(image_path, out_dir, patient_id=None, engine=None):
    os.makedirs(out_dir, exist_ok=True)
    if engine is None:
        engine = ExplainabilityEngine()

    fn = os.path.splitext(os.path.basename(image_path))[0]
    p_id = patient_id or f"PAT-{fn}"
    
    is_idrid = ("IDRiD" in image_path)
    is_drive = ("DRIVE" in image_path or "drive" in fn)
    
    res = engine.run_inference(image_path, is_idrid_domain=is_idrid, is_drive_domain=is_drive)
    
    # 1. HTML Report
    html_content = generate_html_report(res, patient_id=p_id, image_path=image_path)
    html_path = os.path.join(out_dir, f"{p_id}_screening_report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 2. Composite Summary PNG (if not rejected)
    png_path = None
    if res.get('status') != 'NEEDS RECAPTURE':
        composite_bgr = generate_composite_summary_image(res, patient_id=p_id, image_path=image_path)
        png_path = os.path.join(out_dir, f"{p_id}_composite_summary.png")
        cv2.imwrite(png_path, composite_bgr)

    # 3. JSON record
    record = {
        'patient_id': p_id,
        'image_path': image_path,
        'timestamp': datetime.datetime.now().isoformat(),
        'quality': res['quality'],
        'predicted_grade': res.get('predicted_grade'),
        'predicted_label': res.get('predicted_label'),
        'calibrated_confidence': res.get('calibrated_confidence'),
        'referable_score': res.get('referable_score'),
        'referable_status': res.get('referable_status'),
        'optic_disc': res.get('optic_disc_prediction'),
        'fovea': res.get('fovea_prediction'),
        'lesion_summary': res.get('lesion_summary'),
        'vessel_summary': res.get('vessel_summary'),
        'html_report': html_path,
        'composite_png': png_path
    }
    json_path = os.path.join(out_dir, f"{p_id}_record.json")
    with open(json_path, 'w') as f:
        json.dump(record, f, indent=2)

    print(f"Report successfully generated for {p_id}:")
    print(f"  - HTML: {html_path}")
    if png_path:
        print(f"  - PNG:  {png_path}")
    print(f"  - JSON: {json_path}")
    return record

if __name__ == '__main__':
    # Test generation on sample
    print("Testing generate_explainable_report on sample images...")
    root = r"C:\NetraSetu"
    reports_out = os.path.join(root, "results", "final_pipeline", "reports", "sample_screening_reports")
    
    # Pick 3 samples (1 APTOS, 1 IDRiD, 1 DRIVE)
    samples = [
        os.path.join(root, "data", "APTOS", "raw", "train_images", "0024cdab0c1e.png"),
        os.path.join(root, "data", "IDRiD_Grading", "B. Disease Grading", "1. Original Images", "a. Training Set", "IDRiD_002.jpg"),
        os.path.join(root, "data", "DRIVE", "test", "images", "drive_01.png")
    ]
    
    engine = ExplainabilityEngine()
    for s in samples:
        if os.path.exists(s):
            generate_report_for_image(s, reports_out, engine=engine)
