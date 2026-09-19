import re

path = r'C:\NetraSetu\static\js\app.js'
content = open(path, 'r', encoding='utf-8').read()

# Find and replace loadAnalytics function
pattern = re.compile(r'(    async function loadAnalytics\(\) \{.*?\n    \})', re.DOTALL)
m = pattern.search(content)
if not m:
    print("NOT FOUND")
    exit(1)

print("Found at:", m.start(), "-", m.end())
print("First 300 chars:", repr(m.group()[:300]))

new_func = '''    async function loadAnalytics() {
        try {
            const resp = await fetch("/api/analytics");
            if (!resp.ok) throw new Error("Failed to fetch analytics");
            const data = await resp.json();

            // Primary LIVE KPIs (upload+camera only \u2014 never includes sample or demo)
            document.getElementById("anTotalCount").textContent = data.total_screenings.toLocaleString();
            document.getElementById("anAvgLatency").textContent =
                data.avg_processing_time_ms > 0 ? `${data.avg_processing_time_ms} ms` : "\u2014 ms";
            document.getElementById("anPendingReview").textContent = `${data.pending_specialist_reviews} cases`;

            const refRatio = data.total_screenings > 0
                ? (data.referable_cases / data.total_screenings * 100).toFixed(1) : 0;
            document.getElementById("anReferralRatio").textContent = `${refRatio}%`;

            // Source breakdown counters
            const liveEl = document.getElementById("anLiveCount");
            const sampleEl = document.getElementById("anSampleCount");
            const demoEl = document.getElementById("anDemoCount");
            if (liveEl) liveEl.textContent = data.total_screenings.toLocaleString();
            if (sampleEl) sampleEl.textContent = (data.sample_runs || 0).toLocaleString();
            if (demoEl) demoEl.textContent = (data.demo_records || 0).toLocaleString();

            // Refresh disclosures
            applyModeDisclosures(data.data_mode || appMode);

        } catch (e) {
            console.warn("Analytics error:", e);
        }
    }'''

new_content = content[:m.start()] + new_func + content[m.end():]
open(path, 'w', encoding='utf-8').write(new_content)
print("REPLACED OK")
