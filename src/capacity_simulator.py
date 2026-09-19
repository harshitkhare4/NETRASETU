"""
src/capacity_simulator.py
Capacity Planning & Rural Deployment Simulator for NetraSetu.
Smart India Hackathon 2026 | Problem Statement 26038

Models large-scale deployment across Indian Primary Health Centers (PHCs)
and Community Health Centers (CHCs) for 100,000 to 250,000+ patients/year.

Outputs:
- results/final_pipeline/capacity/capacity_model.json
- results/final_pipeline/capacity/capacity_report.txt
- results/final_pipeline/capacity/capacity_plot.png
"""

import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def run_capacity_simulation():
    root = r"C:\NetraSetu"
    out_dir = os.path.join(root, "results", "final_pipeline", "capacity")
    os.makedirs(out_dir, exist_ok=True)

    print("\n========================================================")
    print(" NetraSetu Capacity Planning Simulator (100k - 250k+ Rural Patients)")
    print("========================================================")

    # 1. Component Latency Model (in milliseconds)
    latencies = {
        'Quality Gate Audit': {'gpu': 5.2, 'cpu': 6.8},
        'ResNet-50 Classifier': {'gpu': 24.5, 'cpu': 118.0},
        'Temperature Calibration': {'gpu': 0.4, 'cpu': 0.5},
        'Grad-CAM Attention': {'gpu': 42.1, 'cpu': 195.0},
        'Landmark Localization (OD/FOV)': {'gpu': 19.8, 'cpu': 88.0},
        'Lesion Segmentation (UNet V3)': {'gpu': 64.2, 'cpu': 342.0},
        'Vessel Segmentation (DRIVE UNet)': {'gpu': 21.0, 'cpu': 92.0},
        'Evidence Fusion & Report Assembly': {'gpu': 148.0, 'cpu': 165.0}
    }

    total_gpu_ms = sum(v['gpu'] for v in latencies.values())
    total_cpu_ms = sum(v['cpu'] for v in latencies.values())

    gpu_throughput_sec = 1000.0 / total_gpu_ms
    cpu_throughput_sec = 1000.0 / total_cpu_ms

    gpu_patients_per_hr = gpu_throughput_sec * 3600
    cpu_patients_per_hr = cpu_throughput_sec * 3600

    print(f"Latency Profile:")
    print(f"  - Total GPU Pipeline Latency: {total_gpu_ms:.1f} ms/patient ({gpu_throughput_sec:.2f} patients/sec | {gpu_patients_per_hr:,.0f}/hr)")
    print(f"  - Total CPU Fallback Latency: {total_cpu_ms:.1f} ms/patient ({cpu_throughput_sec:.2f} patients/sec | {cpu_patients_per_hr:,.0f}/hr)")

    # 2. Annual Rural Deployment Scenarios (250 operational days/year, 8 hours/day = 2,000 active hours)
    operating_hours = 2000 # 250 days * 8 hours
    scenarios = [
        {'name': 'District Scale (Target)', 'annual_patients': 100000},
        {'name': 'Multi-District Scale', 'annual_patients': 250000},
        {'name': 'State Screening Program', 'annual_patients': 500000}
    ]

    scenario_results = []
    for sc in scenarios:
        n = sc['annual_patients']
        per_day = n / 250.0
        per_hour = per_day / 8.0
        
        # GPU utilization on 1 single entry workstation (e.g. RTX 3050 / 4060)
        gpu_compute_hours = (n * total_gpu_ms / 1000.0) / 3600.0
        gpu_utilization_pct = (gpu_compute_hours / operating_hours) * 100.0
        
        # Hardware nodes required for <= 50% target utilization
        gpus_needed = max(1, int(np.ceil((gpu_utilization_pct / 50.0))))
        
        scenario_results.append({
            'scenario': sc['name'],
            'annual_patients': n,
            'patients_per_day': round(per_day, 1),
            'patients_per_hour': round(per_hour, 1),
            'gpu_compute_hours_needed': round(gpu_compute_hours, 1),
            'single_gpu_utilization_pct': round(gpu_utilization_pct, 2),
            'recommended_gpu_nodes': gpus_needed
        })

    # 3. Epidemiological Clinical Triage Model (based on Indian rural diabetic retinopathy epidemiology)
    # References: SN-DREAMS, Sankara Nethralaya Rural Studies, AIOS DR guidelines
    epidemiology = {
        'Grade 0 (No DR)': {'prevalence': 0.725, 'referral': False, 'action': 'Annual screening at local PHC'},
        'Grade 1 (Mild DR)': {'prevalence': 0.098, 'referral': False, 'action': '6-12 month follow-up + glycemic control'},
        'Grade 2 (Moderate DR)': {'prevalence': 0.114, 'referral': True, 'action': 'Referral to District Hospital ophthalmologist (within 1 month)'},
        'Grade 3 (Severe DR)': {'prevalence': 0.041, 'referral': True, 'action': 'Urgent referral (within 1-2 weeks)'},
        'Grade 4 (Proliferative DR)': {'prevalence': 0.022, 'referral': True, 'action': 'Immediate vitreoretinal specialist intervention'}
    }

    # For 100k patients:
    patients_100k = 100000
    triage_breakdown = {}
    total_non_referable = 0
    total_referable = 0

    for grade, data in epidemiology.items():
        count = int(round(patients_100k * data['prevalence']))
        triage_breakdown[grade] = {
            'prevalence_pct': data['prevalence'] * 100.0,
            'patients_per_100k': count,
            'referral_required': data['referral'],
            'clinical_action': data['action']
        }
        if data['referral']:
            total_referable += count
        else:
            total_non_referable += count

    workload_reduction_pct = (total_non_referable / patients_100k) * 100.0

    # Quality Gate Recapture Impact
    recapture_rate_pct = 6.4 # ~6.4% of images need recapture
    immediate_recaptures_100k = int(round(patients_100k * (recapture_rate_pct / 100.0)))
    estimated_travel_cost_saved_inr = immediate_recaptures_100k * 450 # Avg rural patient transit + day wage ~ 450 INR

    print(f"\nClinical Triage Impact (per 100,000 patients):")
    print(f"  - Non-Referable (Screened & Managed at PHC): {total_non_referable:,} ({workload_reduction_pct:.1f}%)")
    print(f"  - Referable (Escalated to Specialist Care):  {total_referable:,} ({100 - workload_reduction_pct:.1f}%)")
    print(f"  - Ophthalmologist Workload Reduction:       {workload_reduction_pct:.1f}%")
    print(f"  - Immediate Recaptures Prevented at Point-of-Care: {immediate_recaptures_100k:,} patients")
    print(f"  - Patient Travel/Wage Loss Savings:         Rs. {estimated_travel_cost_saved_inr:,} INR")

    # 4. Save JSON Model
    model_json = {
        'timestamp': '2026-09-19T23:30:00IST',
        'pipeline_latency_ms': latencies,
        'total_gpu_latency_ms': round(total_gpu_ms, 2),
        'total_cpu_latency_ms': round(total_cpu_ms, 2),
        'throughput_per_sec': {
            'gpu': round(gpu_throughput_sec, 2),
            'cpu': round(cpu_throughput_sec, 2)
        },
        'scaling_scenarios': scenario_results,
        'epidemiological_triage_100k': {
            'total_screened': patients_100k,
            'non_referable_managed_at_phc': total_non_referable,
            'referable_escalated_to_specialist': total_referable,
            'workload_reduction_pct': round(workload_reduction_pct, 2),
            'grades': triage_breakdown,
            'point_of_care_recaptures_saved': immediate_recaptures_100k,
            'economic_savings_inr': estimated_travel_cost_saved_inr
        }
    }

    json_path = os.path.join(out_dir, "capacity_model.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(model_json, f, indent=2)

    # 5. Generate Text Report
    report_text = f"""================================================================================
NETRASETU CAPACITY PLANNING & RURAL DEPLOYMENT REPORT
Smart India Hackathon 2026 | Problem Statement ID: 26038
Project: NetraSetu — Explainable AI for Diabetic Retinopathy Screening in Rural India
================================================================================

1. EXECUTIVE SUMMARY:
NetraSetu has been architected to address the acute shortage of vitreoretinal specialists 
in rural India (where the doctor-to-patient ratio exceeds 1:100,000 in rural districts).
By combining an automated Image Quality Gate, calibrated ResNet-50 classification, 
lesion/vessel segmentation, and multi-modal explainability reports, NetraSetu enables 
accredited healthcare workers (ASHA/ANM workers and PHC medical officers) to triage 
retinal screenings locally with sub-second turnaround time.

2. COMPONENT LATENCY BREAKDOWN (Single Patient Inference):
--------------------------------------------------------------------------------
Pipeline Component                     | GPU Latency (ms) | CPU Fallback (ms)
--------------------------------------------------------------------------------
1. Quality Gate Audit                  | {latencies['Quality Gate Audit']['gpu']:>16.1f} | {latencies['Quality Gate Audit']['cpu']:>17.1f}
2. ResNet-50 Classifier (Baseline)    | {latencies['ResNet-50 Classifier']['gpu']:>16.1f} | {latencies['ResNet-50 Classifier']['cpu']:>17.1f}
3. Temperature Scaling Calibration     | {latencies['Temperature Calibration']['gpu']:>16.1f} | {latencies['Temperature Calibration']['cpu']:>17.1f}
4. Grad-CAM Visual Attention           | {latencies['Grad-CAM Attention']['gpu']:>16.1f} | {latencies['Grad-CAM Attention']['cpu']:>17.1f}
5. Landmark Localization (OD/FOV)      | {latencies['Landmark Localization (OD/FOV)']['gpu']:>16.1f} | {latencies['Landmark Localization (OD/FOV)']['cpu']:>17.1f}
6. Lesion Segmentation (UNet V3)       | {latencies['Lesion Segmentation (UNet V3)']['gpu']:>16.1f} | {latencies['Lesion Segmentation (UNet V3)']['cpu']:>17.1f}
7. Retinal Vessel Segmentation (DRIVE) | {latencies['Vessel Segmentation (DRIVE UNet)']['gpu']:>16.1f} | {latencies['Vessel Segmentation (DRIVE UNet)']['cpu']:>17.1f}
8. Evidence Fusion & Report Assembly   | {latencies['Evidence Fusion & Report Assembly']['gpu']:>16.1f} | {latencies['Evidence Fusion & Report Assembly']['cpu']:>17.1f}
--------------------------------------------------------------------------------
TOTAL PIPELINE LATENCY                 | {total_gpu_ms:>16.1f} ms | {total_cpu_ms:>16.1f} ms
THROUGHPUT                             | {gpu_throughput_sec:>14.2f} pts/sec | {cpu_throughput_sec:>15.2f} pts/sec
HOURLY CAPACITY                        | {gpu_patients_per_hr:>14,.0f} pts/hr  | {cpu_patients_per_hr:>15,.0f} pts/hr
--------------------------------------------------------------------------------

3. SCALABILITY & INFRASTRUCTURE SIZING (2,000 Operating Hours/Year):
--------------------------------------------------------------------------------
Scale Scenario         | Patients/Yr | Pts/Day | Pts/Hr | Nominal Modeled Utilization | Required Nodes
--------------------------------------------------------------------------------
District Scale         |     100,000 |   400.0 |   50.0 |                       0.45% | 1 Edge Workstation
Multi-District Scale   |     250,000 | 1,000.0 |  125.0 |                       1.13% | 1 Edge Workstation / Cloud
State Screening Prog.  |     500,000 | 2,000.0 |  250.0 |                       2.26% | 1-2 Cloud GPU Instances
--------------------------------------------------------------------------------
Key Engineering Finding: A single standard workstation with an entry-level GPU operates
at approximately 0.45% nominal modeled compute utilization to process 100,000 screenings annually.

ENGINEERING DISCLOSURE:
These figures are parametric engineering estimates based on configured timing assumptions
and do not represent measured production deployment capacity.

4. CLINICAL WORKLOAD REDUCTION & TRIAGE FUNNEL (Illustrative Scenario-Based Planning Estimate):
[Assumed Population Case Mix: 72.5% Grade 0, 9.8% Grade 1, 11.4% Grade 2, 4.1% Grade 3, 2.2% Grade 4]
(Note: These are illustrative scenario-based planning estimates, not observed clinical deployment data)
- Total Patients Screened:                    100,000 (100.0%)
- Non-Referable Patients (Grade 0: No DR):     72,500 (72.5%) -> Modeled for PHC routine follow-up
- Low-Risk Early DR (Grade 1: Mild DR):         9,800  (9.8%) -> Modeled for 1-year follow-up
- Subtotal Non-Referable (Routine Filtered):   82,300 (82.3%)
- Referable Moderate DR (Grade 2):             11,400 (11.4%) -> Modeled for specialist referral
- Severe Non-Proliferative DR (Grade 3):        4,100  (4.1%) -> Modeled for priority referral
- Proliferative DR (Grade 4):                   2,200  (2.2%) -> Modeled for urgent referral
- Total Referrals Escalated to Specialist:     17,700 (17.7%)

OPHTHALMOLOGIST WORKLOAD REDUCTION: Modeled 82.3% routine filtering under the assumed epidemiological
case mix, allowing specialist consultation capacity to prioritize referable pathology.

5. ECONOMIC & ACCESS IMPACT (Illustrative Scenario-Based Planning Estimate):
[Assumed Optical Recapture Rate: 6.4% | Assumed Transit + Day Wage Savings: Rs. 450 INR per patient]
- Immediate Point-of-Care Recaptures: 6,400 patients (6.4%) alerted to blurry/dark images
  while still present at the screening station, eliminating estimated return journeys.
- Estimated Direct Travel & Wage Loss Saved: Rs. 2,880,000 INR per 100,000 screened patients
  (illustrative planning estimate; actual savings depend on local transit and regional wage scales).
- Modeled Compute Latency: Sub-second turnaround (325.2 ms on GPU).

================================================================================
End of Capacity Planning Report
================================================================================
"""
    txt_path = os.path.join(out_dir, "capacity_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 6. Generate Multi-Panel Publication-Quality Visualization Plot
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("NetraSetu — 100k+ Patient Rural Screening Capacity & Triage Simulator\nParametric Engineering Model | Smart India Hackathon 2026 | PS 26038", fontsize=14, fontweight='bold', y=0.98)

    # Subplot 1: Latency Breakdown (Horizontal Bar)
    ax1 = axes[0, 0]
    names = list(latencies.keys())
    gpu_vals = [latencies[k]['gpu'] for k in names]
    cpu_vals = [latencies[k]['cpu'] for k in names]
    y_pos = np.arange(len(names))
    bar_h = 0.35

    ax1.barh(y_pos - bar_h/2, gpu_vals, bar_h, label='GPU (RTX 3050)', color='#0284c7')
    ax1.barh(y_pos + bar_h/2, cpu_vals, bar_h, label='CPU Fallback', color='#94a3b8')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Latency per Patient (ms)', fontweight='bold')
    ax1.set_title(f'Component Latency Breakdown (Total GPU: {total_gpu_ms:.1f}ms)', fontweight='bold', fontsize=11)
    ax1.legend()
    for i, v in enumerate(gpu_vals):
        ax1.text(v + 3, i - bar_h/2 + 0.1, f"{v:.1f}ms", fontsize=8, color='#0369a1', fontweight='bold')

    # Subplot 2: Triage Referral Funnel (Donut Chart)
    ax2 = axes[0, 1]
    labels = ['No DR (PHC)', 'Mild DR (Follow-up)', 'Moderate DR (Refer)', 'Severe DR (Refer)', 'PDR (Urgent)']
    sizes = [72.5, 9.8, 11.4, 4.1, 2.2]
    colors = ['#10b981', '#34d399', '#f59e0b', '#ef4444', '#b91c1c']
    explode = (0.02, 0.02, 0.05, 0.08, 0.1)

    wedges, texts, autotexts = ax2.pie(
        sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        startangle=140, colors=colors, textprops={'fontsize': 9},
        pctdistance=0.8
    )
    # Donut hole
    centre_circle = plt.Circle((0,0), 0.55, fc='white')
    ax2.add_artist(centre_circle)
    ax2.set_title("Clinical Screening Triage Distribution\n(82.3% Filtered at PHC | 17.7% Specialist Referral)", fontweight='bold', fontsize=11)

    # Subplot 3: Annual Patient Scale vs GPU Utilization
    ax3 = axes[1, 0]
    sc_names = [s['scenario'] for s in scenario_results]
    pts = [s['annual_patients'] / 1000.0 for s in scenario_results]
    utils = [s['single_gpu_utilization_pct'] for s in scenario_results]

    ax3_twin = ax3.twinx()
    bars = ax3.bar(sc_names, pts, width=0.4, color='#3b82f6', alpha=0.85, label='Patients (k/year)')
    line = ax3_twin.plot(sc_names, utils, color='#dc2626', marker='o', linewidth=2.5, label='Single GPU Util (%)')

    ax3.set_ylabel('Annual Screenings (Thousands)', color='#1d4ed8', fontweight='bold')
    ax3_twin.set_ylabel('Nominal Modeled Utilization (%)', color='#b91c1c', fontweight='bold')
    ax3_twin.set_ylim(0, 5)
    ax3.set_title('Screening Volume vs Modeled Utilization (2,000 hrs/yr)', fontweight='bold', fontsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval)}k", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for x_i, u in enumerate(utils):
        ax3_twin.text(x_i, u + 0.25, f"{u:.2f}%", ha='center', color='#b91c1c', fontweight='bold', fontsize=9)

    # Subplot 4: Specialist Time & Economic Savings
    ax4 = axes[1, 1]
    metrics = ['Ophthalmologist\nWorkload Reduction', 'Immediate Point-of-Care\nRecapture Rate', 'Avg Turnaround\nAcceleration']
    values = [82.3, 6.4, 99.8] # 99.8% acceleration from weeks to seconds
    colors_m = ['#10b981', '#06b6d4', '#6366f1']
    
    bars_m = ax4.bar(metrics, values, color=colors_m, width=0.5)
    ax4.set_ylim(0, 115)
    ax4.set_ylabel('Impact Percentage (%)', fontweight='bold')
    ax4.set_title('Rural Healthcare Delivery Impact Metrics', fontweight='bold', fontsize=11)
    for b in bars_m:
        h = b.get_height()
        ax4.text(b.get_x() + b.get_width()/2.0, h + 2, f"{h:.1f}%", ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "capacity_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCapacity Planning Simulation Completed Successfully!")
    print(f"  - Model:  {json_path}")
    print(f"  - Report: {txt_path}")
    print(f"  - Plot:   {plot_path}")
    return model_json

if __name__ == '__main__':
    run_capacity_simulation()
