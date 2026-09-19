/**
 * NETRASETU - HEALTHCARE AI SCREENING PLATFORM CONTROLLER
 * Smart India Hackathon 2026 - Problem Statement 26038
 * Strict Light-Theme Architecture
 */

document.addEventListener("DOMContentLoaded", () => {
    // Application State
    let activeTab = "homeTab";
    let selectedFile = null;
    let selectedSampleId = null;
    let currentResultData = null;
    let activeMediaStream = null;
    let capturedDataUrl = null;

    // Navigation Tabs
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Top Status
    const systemStatusBadge = document.getElementById("systemStatusBadge");
    const systemStatusText = document.getElementById("systemStatusText");
    const navReviewCountBadge = document.getElementById("navReviewCountBadge");

    // Home CTAs
    const btnHomeStartScreening = document.getElementById("btnHomeStartScreening");
    const btnHomeStartCamera = document.getElementById("btnHomeStartCamera");
    const btnHomeExploreServices = document.getElementById("btnHomeExploreServices");

    // Screening Mode Switchers
    const btnModeUpload = document.getElementById("btnModeUpload");
    const btnModeCamera = document.getElementById("btnModeCamera");
    const uploadCardSection = document.getElementById("uploadCardSection");
    const cameraCardSection = document.getElementById("cameraCardSection");

    // Upload Mode Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const dropzonePreview = document.getElementById("dropzonePreview");
    const previewImage = document.getElementById("previewImage");
    const previewFilename = document.getElementById("previewFilename");
    const btnChangeImage = document.getElementById("btnChangeImage");
    const sampleChips = document.querySelectorAll(".sample-chip");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnAnalyzeText = document.getElementById("btnAnalyzeText");
    const ctaHint = document.getElementById("ctaHint");

    // Camera Mode Elements
    const webcamVideo = document.getElementById("webcamVideo");
    const webcamCanvas = document.getElementById("webcamCanvas");
    const cameraReticle = document.getElementById("cameraReticle");
    const cameraCapturedView = document.getElementById("cameraCapturedView");
    const cameraCapturedImg = document.getElementById("cameraCapturedImg");
    const cameraDot = document.getElementById("cameraDot");
    const cameraStatusText = document.getElementById("cameraStatusText");
    const btnStartCamera = document.getElementById("btnStartCamera");
    const btnCaptureFrame = document.getElementById("btnCaptureFrame");
    const btnRetakeFrame = document.getElementById("btnRetakeFrame");
    const btnAnalyzeCameraCapture = document.getElementById("btnAnalyzeCameraCapture");
    const camSpinner = document.getElementById("camSpinner");
    const btnAnalyzeCameraText = document.getElementById("btnAnalyzeCameraText");

    // Screening Results Elements
    const resultsContainer = document.getElementById("resultsContainer");
    const qualityBadge = document.getElementById("qualityBadge");
    const qualityStatusText = document.getElementById("qualityStatusText");
    const blurScoreVal = document.getElementById("blurScoreVal");
    const brightnessVal = document.getElementById("brightnessVal");
    const contrastVal = document.getElementById("contrastVal");
    const fovVal = document.getElementById("fovVal");
    const qualityAlertBox = document.getElementById("qualityAlertBox");
    const recaptureAdviceText = document.getElementById("recaptureAdviceText");
    const preprocessingNoteText = document.getElementById("preprocessingNoteText");

    const drLevelText = document.getElementById("drLevelText");
    const drSeverityName = document.getElementById("drSeverityName");
    const severityHeroBadge = document.getElementById("severityHeroBadge");
    const icdrGradeText = document.getElementById("icdrGradeText");
    const classConfidenceText = document.getElementById("classConfidenceText");
    const probabilityBarsList = document.getElementById("probabilityBarsList");

    const referableStatusHero = document.getElementById("referableStatusHero");
    const referableStatusText = document.getElementById("referableStatusText");
    const referableStatusDesc = document.getElementById("referableStatusDesc");
    const referableScoreVal = document.getElementById("referableScoreVal");
    const referableGaugeFill = document.getElementById("referableGaugeFill");
    const referralDetailText = document.getElementById("referralDetailText");

    const enhancedResultImg = document.getElementById("enhancedResultImg");
    const gradcamResultImg = document.getElementById("gradcamResultImg");
    const btnToggleSideBySide = document.getElementById("btnToggleSideBySide");
    const btnViewLargerModal = document.getElementById("btnViewLargerModal");
    const comparisonGrid = document.getElementById("comparisonGrid");

    const lesionAccordionToggle = document.getElementById("lesionAccordionToggle");
    const lesionAccordionBody = document.getElementById("lesionAccordionBody");
    const accordionArrow = document.getElementById("accordionArrow");
    const btnRunLesionAnalysis = document.getElementById("btnRunLesionAnalysis");
    const lesionSpinner = document.getElementById("lesionSpinner");
    const btnRunLesionText = document.getElementById("btnRunLesionText");
    const lesionResultsView = document.getElementById("lesionResultsView");
    const lesionsOverlayImg = document.getElementById("lesionsOverlayImg");
    const lesionStatMA = document.getElementById("lesionStatMA");
    const lesionStatHEM = document.getElementById("lesionStatHEM");
    const lesionStatEX = document.getElementById("lesionStatEX");
    const lesionStatSE = document.getElementById("lesionStatSE");

    const reportDateText = document.getElementById("reportDateText");
    const reportCaseIdText = document.getElementById("reportCaseIdText");
    const reportImageIdText = document.getElementById("reportImageIdText");
    const repQualityStatus = document.getElementById("repQualityStatus");
    const repBlurScore = document.getElementById("repBlurScore");
    const repBrightness = document.getElementById("repBrightness");
    const repContrast = document.getElementById("repContrast");
    const repFov = document.getElementById("repFov");
    const repAlerts = document.getElementById("repAlerts");
    const repIcdrGrade = document.getElementById("repIcdrGrade");
    const repSeverity = document.getElementById("repSeverity");
    const repConfidence = document.getElementById("repConfidence");
    const repProbSummary = document.getElementById("repProbSummary");
    const repReferableScore = document.getElementById("repReferableScore");
    const repReferableStatus = document.getElementById("repReferableStatus");
    const repRecommendation = document.getElementById("repRecommendation");

    const btnDownloadReportTxt = document.getElementById("btnDownloadReportTxt");
    const btnDownloadJson = document.getElementById("btnDownloadJson");
    const btnPrintReport = document.getElementById("btnPrintReport");

    // Reports Subnav & Views
    const btnSubnavHistory = document.getElementById("btnSubnavHistory");
    const btnSubnavMonthly = document.getElementById("btnSubnavMonthly");
    const btnSubnavYearly = document.getElementById("btnSubnavYearly");
    const subviewHistory = document.getElementById("subviewHistory");
    const subviewMonthly = document.getElementById("subviewMonthly");
    const subviewYearly = document.getElementById("subviewYearly");
    const historyTableBody = document.getElementById("historyTableBody");
    const historySearchInput = document.getElementById("historySearchInput");
    const filterPills = document.querySelectorAll(".filter-pill[data-filter]");
    const btnExportCsv = document.getElementById("btnExportCsv");
    const monthSelect = document.getElementById("monthSelect");

    // Review Queue Elements
    const reviewTableBody = document.getElementById("reviewTableBody");
    const reviewFilterPills = document.querySelectorAll(".filter-pill[data-review-filter]");
    const reviewModal = document.getElementById("reviewModal");
    const reviewModalBackdrop = document.getElementById("reviewModalBackdrop");
    const btnCloseReviewModal = document.getElementById("btnCloseReviewModal");
    const btnCancelReviewModal = document.getElementById("btnCancelReviewModal");
    const btnSaveReviewModal = document.getElementById("btnSaveReviewModal");
    const revModalCaseId = document.getElementById("revModalCaseId");
    const revModalGrade = document.getElementById("revModalGrade");
    const revModalScore = document.getElementById("revModalScore");
    const revModalStatusSelect = document.getElementById("revModalStatusSelect");
    const revModalNotes = document.getElementById("revModalNotes");
    let currentReviewCaseId = null;

    // Capacity Planner Elements
    const inputTargetPatients = document.getElementById("inputTargetPatients");
    const inputWorkingDays = document.getElementById("inputWorkingDays");
    const inputClinics = document.getElementById("inputClinics");
    const inputLatency = document.getElementById("inputLatency");
    const calcReqDay = document.getElementById("calcReqDay");
    const calcReqClinic = document.getElementById("calcReqClinic");
    const calcAiCapDay = document.getElementById("calcAiCapDay");
    const calcSpecReviews = document.getElementById("calcSpecReviews");

    // Mode Indicator Elements
    const modeIndicator = document.getElementById("modeIndicator");
    const modeDot = document.getElementById("modeDot");
    const modeLabelText = document.getElementById("modeLabelText");
    const btnResetDemoData = document.getElementById("btnResetDemoData");
    const reportsModeDisclosure = document.getElementById("reportsModeDisclosure");
    const reportsLiveNotice = document.getElementById("reportsLiveNotice");
    const analyticsModeDisclosure = document.getElementById("analyticsModeDisclosure");
    const analyticsLiveNotice = document.getElementById("analyticsLiveNotice");

    // Application-level mode state (populated from /api/mode)
    let appMode = "live"; // default until /api/mode responds

    // Modals Lightbox
    const imageModal = document.getElementById("imageModal");
    const modalBackdrop = document.getElementById("modalBackdrop");
    const btnCloseModal = document.getElementById("btnCloseModal");
    const modalEnhancedImg = document.getElementById("modalEnhancedImg");
    const modalGradcamImg = document.getElementById("modalGradcamImg");

    // ============================================================
    // 1. SYSTEM INITIALIZATION & HEALTH TELEMETRY
    // ============================================================

    async function checkSystemHealth() {
        try {
            const resp = await fetch("/api/health");
            if (!resp.ok) throw new Error("Health check returned status " + resp.status);
            const data = await resp.json();

            if (data.status === "ready" && data.classifier_loaded) {
                const deviceLabel = data.cuda_available ? `CUDA • ${data.gpu}` : "CPU Mode";
                systemStatusText.textContent = `Model Ready (${deviceLabel})`;
                systemStatusBadge.style.color = "#0d9488";
                systemStatusBadge.style.background = "#f0fdfa";
                systemStatusBadge.style.borderColor = "#99f6e4";

                const statDevice = document.getElementById("statDevice");
                const statGpuName = document.getElementById("statGpuName");
                if (statDevice) statDevice.textContent = data.device.toUpperCase();
                if (statGpuName) statGpuName.textContent = data.gpu;
            }
        } catch (err) {
            console.warn("Health check error:", err);
            systemStatusText.textContent = "Service Offline";
        }
    }

    async function updateReviewQueueCount() {
        try {
            const resp = await fetch("/api/review-queue?status=Pending Review");
            if (resp.ok) {
                const data = await resp.json();
                if (navReviewCountBadge) {
                    navReviewCountBadge.textContent = data.count;
                    navReviewCountBadge.style.display = data.count > 0 ? "inline-block" : "none";
                }
            }
        } catch (e) {
            console.warn("Could not fetch review count:", e);
        }
    }

    checkSystemHealth();
    updateReviewQueueCount();
    initMode();

    // ============================================================
    // 1b. LIVE / DEMO MODE INITIALIZATION
    // ============================================================

    async function initMode() {
        try {
            const resp = await fetch("/api/mode");
            if (!resp.ok) return;
            const data = await resp.json();
            appMode = data.mode; // "live" or "demo"

            if (modeLabelText) modeLabelText.textContent = data.label;

            if (modeDot) {
                modeDot.style.background = data.mode === "live" ? "#16a34a" : "#f59e0b";
            }
            if (modeIndicator) {
                modeIndicator.style.color = data.mode === "live" ? "#166534" : "#92400e";
                modeIndicator.style.background = data.mode === "live" ? "#f0fdf4" : "#fef3c7";
                modeIndicator.style.borderColor = data.mode === "live" ? "#86efac" : "#fcd34d";
            }

            // Show Reset Demo Data button only in DEMO MODE
            if (btnResetDemoData) {
                btnResetDemoData.style.display = data.mode === "demo" ? "inline-block" : "none";
            }

            // Toggle disclosure banners
            applyModeDisclosures(data.mode);

        } catch (e) {
            console.warn("Could not fetch app mode:", e);
        }
    }

    function applyModeDisclosures(mode) {
        const isDemo = mode === "demo";
        if (reportsModeDisclosure) reportsModeDisclosure.style.display = isDemo ? "flex" : "none";
        if (reportsLiveNotice) reportsLiveNotice.style.display = isDemo ? "none" : "flex";
        if (analyticsModeDisclosure) analyticsModeDisclosure.style.display = isDemo ? "flex" : "none";
        if (analyticsLiveNotice) analyticsLiveNotice.style.display = isDemo ? "none" : "flex";
    }

    if (btnResetDemoData) {
        btnResetDemoData.addEventListener("click", async () => {
            if (!confirm(
                "Remove all synthetic/seeded demo records from the database?\n\n" +
                "Live screening activity and test samples will NOT be deleted."
            )) return;

            try {
                btnResetDemoData.disabled = true;
                btnResetDemoData.textContent = "Removing...";
                const resp = await fetch("/api/reset-demo-data", { method: "POST" });
                const result = await resp.json();
                alert(`Done. Removed ${result.total_deleted} synthetic records.\n" +
                    "(${result.deleted_seeded} old-seeder + ${result.deleted_demo} demo-type)`);
                // Reload current view
                loadScreeningHistory();
                loadAnalytics();
                updateReviewQueueCount();
            } catch (e) {
                alert("Reset failed: " + e.message);
            } finally {
                btnResetDemoData.disabled = false;
                btnResetDemoData.textContent = "Reset Demo Data";
            }
        });
    }

    // ============================================================
    // 2. PRIMARY TAB ROUTER
    // ============================================================

    function switchTab(targetTabId) {
        activeTab = targetTabId;

        navTabs.forEach((t) => t.classList.remove("active"));
        tabPanes.forEach((p) => {
            p.classList.remove("active");
            p.style.display = "none";
        });

        const activeNavBtn = document.querySelector(`.nav-tab[data-tab='${targetTabId}']`);
        if (activeNavBtn) activeNavBtn.classList.add("active");

        const targetPane = document.getElementById(targetTabId);
        if (targetPane) {
            targetPane.classList.add("active");
            targetPane.style.display = "block";
            window.scrollTo({ top: 0, behavior: "smooth" });
        }

        // Lazy-load data when switching to specialized views
        if (targetTabId === "reportsTab") {
            loadScreeningHistory();
            loadMonthlyReport();
        } else if (targetTabId === "reviewTab") {
            loadReviewQueue();
        } else if (targetTabId === "analyticsTab") {
            loadAnalytics();
            updateCapacityCalculations();
        }
    }

    navTabs.forEach((btn) => {
        btn.addEventListener("click", () => {
            switchTab(btn.getAttribute("data-tab"));
        });
    });

    // Home CTAs
    if (btnHomeStartScreening) {
        btnHomeStartScreening.addEventListener("click", () => {
            switchTab("screeningTab");
            setScreeningMode("upload");
        });
    }

    if (btnHomeStartCamera) {
        btnHomeStartCamera.addEventListener("click", () => {
            switchTab("screeningTab");
            setScreeningMode("camera");
        });
    }

    if (btnHomeExploreServices) {
        btnHomeExploreServices.addEventListener("click", () => {
            switchTab("servicesTab");
        });
    }

    // ============================================================
    // 3. SCREENING MODE TOGGLE (UPLOAD VS LIVE CAMERA)
    // ============================================================

    function setScreeningMode(mode) {
        if (mode === "upload") {
            btnModeUpload.classList.add("active");
            btnModeCamera.classList.remove("active");
            uploadCardSection.style.display = "block";
            cameraCardSection.style.display = "none";
            stopCameraStream();
        } else {
            btnModeCamera.classList.add("active");
            btnModeUpload.classList.remove("active");
            cameraCardSection.style.display = "block";
            uploadCardSection.style.display = "none";
        }
    }

    btnModeUpload.addEventListener("click", () => setScreeningMode("upload"));
    btnModeCamera.addEventListener("click", () => setScreeningMode("camera"));

    // ============================================================
    // 4. LIVE CAMERA IMPLEMENTATION (BROWSER WEBCAM)
    // ============================================================

    async function startCameraStream() {
        try {
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                alert("Webcam access is not supported by your current browser environment.");
                return;
            }

            activeMediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 } },
                audio: false
            });

            webcamVideo.srcObject = activeMediaStream;
            btnCaptureFrame.disabled = false;
            btnStartCamera.style.display = "none";
            btnRetakeFrame.style.display = "none";
            btnAnalyzeCameraCapture.style.display = "none";
            cameraCapturedView.style.display = "none";
            cameraReticle.style.display = "flex";

            cameraStatusText.textContent = "Camera Active (Streaming)";
            cameraDot.style.backgroundColor = "#10b981";
        } catch (err) {
            console.error("Camera access error:", err);
            alert("Camera Error: Unable to access video input. Please ensure permissions are granted.");
            cameraStatusText.textContent = "Camera Error / Denied";
            cameraDot.style.backgroundColor = "#ef4444";
        }
    }

    function stopCameraStream() {
        if (activeMediaStream) {
            activeMediaStream.getTracks().forEach((track) => track.stop());
            activeMediaStream = null;
        }
        if (webcamVideo) webcamVideo.srcObject = null;
        btnStartCamera.style.display = "inline-flex";
        btnCaptureFrame.disabled = true;
        cameraStatusText.textContent = "Camera Disconnected";
        cameraDot.style.backgroundColor = "#64748b";
    }

    function captureFrame() {
        if (!webcamVideo || webcamVideo.videoWidth === 0) return;

        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const ctx = webcamCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        capturedDataUrl = webcamCanvas.toDataURL("image/jpeg", 0.95);
        cameraCapturedImg.src = capturedDataUrl;

        cameraCapturedView.style.display = "flex";
        cameraReticle.style.display = "none";
        btnCaptureFrame.style.display = "none";
        btnRetakeFrame.style.display = "inline-flex";
        btnAnalyzeCameraCapture.style.display = "inline-flex";
        cameraStatusText.textContent = "Frame Captured";
    }

    function retakeFrame() {
        capturedDataUrl = null;
        cameraCapturedView.style.display = "none";
        cameraReticle.style.display = "flex";
        btnCaptureFrame.style.display = "inline-flex";
        btnCaptureFrame.disabled = false;
        btnRetakeFrame.style.display = "none";
        btnAnalyzeCameraCapture.style.display = "none";
        cameraStatusText.textContent = "Camera Active (Streaming)";
    }

    btnStartCamera.addEventListener("click", startCameraStream);
    btnCaptureFrame.addEventListener("click", captureFrame);
    btnRetakeFrame.addEventListener("click", retakeFrame);

    btnAnalyzeCameraCapture.addEventListener("click", () => {
        executeScreening(false);
    });

    // ============================================================
    // 5. UPLOAD & DEMO SAMPLES HANDLING
    // ============================================================

    dropzone.addEventListener("click", (e) => {
        if (e.target !== btnChangeImage) fileInput.click();
    });

    btnChangeImage.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        fileInput.click();
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) handleFileSelection(dt.files[0]);
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) handleFileSelection(e.target.files[0]);
    });

    function handleFileSelection(file) {
        if (!file.name.match(/\.(jpg|jpeg|png)$/i)) {
            alert("Invalid file format. Please upload JPG or PNG.");
            return;
        }

        selectedFile = file;
        selectedSampleId = null;
        sampleChips.forEach((chip) => chip.classList.remove("active"));

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewFilename.textContent = file.name;
            dropzonePrompt.style.display = "none";
            dropzonePreview.style.display = "flex";
        };
        reader.readAsDataURL(file);

        btnAnalyze.disabled = false;
        ctaHint.textContent = `Ready to analyze: ${file.name}`;
    }

    sampleChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            const sampleId = chip.getAttribute("data-sample-id");
            selectedSampleId = sampleId;
            selectedFile = null;

            sampleChips.forEach((c) => c.classList.remove("active"));
            chip.classList.add("active");

            const title = chip.querySelector("strong").textContent;
            dropzonePrompt.style.display = "none";
            dropzonePreview.style.display = "flex";
            previewImage.src = "";
            previewFilename.textContent = `Sample: ${title}`;

            btnAnalyze.disabled = false;
            ctaHint.textContent = `Selected demo sample: ${title}`;
        });
    });

    async function executeScreening(override_quality = false) {
        if (!selectedFile && !selectedSampleId && !capturedDataUrl) {
            alert("Please select or upload a retinal fundus photograph first.");
            return;
        }

        setLoadingState(true);

        try {
            let resp;
            if (activeTab === "screeningTab" && cameraCardSection && cameraCardSection.style.display === "block" && capturedDataUrl) {
                resp = await fetch("/api/analyze-camera", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image_data: capturedDataUrl, override_quality: override_quality })
                });
            } else if (selectedFile) {
                const formData = new FormData();
                formData.append("file", selectedFile);
                if (override_quality) formData.append("override_quality", "true");
                resp = await fetch("/api/analyze", { method: "POST", body: formData });
            } else if (selectedSampleId) {
                const url = `/api/sample/${selectedSampleId}` + (override_quality ? "?override_quality=true" : "");
                resp = await fetch(url, { method: "POST" });
            }

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.error || errData.message || "Screening analysis failed.");
            }

            const data = await resp.json();
            currentResultData = data;
            renderResults(data);
            updateReviewQueueCount();

            resultsContainer.style.display = "block";
            resultsContainer.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (err) {
            console.error("Screening error:", err);
            alert("Screening Error: " + err.message);
        } finally {
            setLoadingState(false);
        }
    }

    btnAnalyze.addEventListener("click", () => executeScreening(false));

    const btnAnalyzeAnyway = document.getElementById("btnAnalyzeAnyway");
    if (btnAnalyzeAnyway) {
        btnAnalyzeAnyway.addEventListener("click", () => executeScreening(true));
    }

    const btnRetakeFromRejection = document.getElementById("btnRetakeFromRejection");
    if (btnRetakeFromRejection) {
        btnRetakeFromRejection.addEventListener("click", () => {
            resultsContainer.style.display = "none";
            window.scrollTo({ top: 300, behavior: "smooth" });
        });
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            btnAnalyze.disabled = true;
            btnSpinner.style.display = "inline-block";
            btnAnalyzeText.textContent = "Analyzing Retina...";
            ctaHint.textContent = "Processing: Quality Gate Check → ResNet-50 → Grad-CAM...";
        } else {
            btnAnalyze.disabled = false;
            btnSpinner.style.display = "none";
            btnAnalyzeText.textContent = "Analyze Retina";
            ctaHint.textContent = "Analysis complete. Results displayed below.";
        }
    }

    // ============================================================
    // 6. RENDER ANALYSIS RESULTS DYNAMICALLY
    // ============================================================

    function renderResults(data) {
        // Step 1: Quality Triage
        const q = data.quality;
        qualityBadge.className = `quality-status-badge ${q.status.toLowerCase()}`;
        qualityStatusText.textContent = q.status === "GOOD" ? "GOOD ✓" : "REVIEW ⚠";

        blurScoreVal.textContent = q.blur_score;
        brightnessVal.textContent = q.brightness;
        contrastVal.textContent = q.contrast;
        fovVal.textContent = `${(q.retinal_area_ratio * 100).toFixed(1)}%`;

        if (q.status === "REVIEW" && data.recapture_advice) {
            qualityAlertBox.style.display = "flex";
            recaptureAdviceText.textContent = data.recapture_advice;
        } else {
            qualityAlertBox.style.display = "none";
        }

        preprocessingNoteText.textContent = data.preprocessing_note || "Safe preprocessing pipeline executed.";

        const qualityRejectionCard = document.getElementById("qualityRejectionCard");
        const overrideWarningBanner = document.getElementById("overrideWarningBanner");
        const coreResultsGrid = document.getElementById("coreResultsGrid");
        const explainabilityStudioCard = document.querySelector(".explain-studio-card") || document.getElementById("comparisonGrid")?.closest(".card");

        // QUALITY GATE REJECTION FLOW
        if (data.quality_gate_passed === false && !data.is_research_override) {
            if (qualityRejectionCard) {
                qualityRejectionCard.style.display = "block";
                const reasonsList = document.getElementById("rejectionReasonsList");
                if (reasonsList) reasonsList.textContent = (q.reasons && q.reasons.length > 0) ? q.reasons.join(", ") : "Blur or FOV criteria failed";
                const adviceText = document.getElementById("rejectionAdviceText");
                if (adviceText) adviceText.textContent = data.recapture_advice || "Please recapture with centered focus and stable illumination.";
            }
            if (overrideWarningBanner) overrideWarningBanner.style.display = "none";
            if (coreResultsGrid) coreResultsGrid.style.display = "none";
            if (explainabilityStudioCard) explainabilityStudioCard.style.display = "none";

            // Update Report Paper for Quality Gate Rejection
            reportDateText.textContent = `Date: ${data.timestamp}`;
            if (reportCaseIdText) reportCaseIdText.textContent = `Case ID: ${data.case_id}`;
            reportImageIdText.textContent = `Image: ${data.image}`;

            repQualityStatus.textContent = q.status + " (HALTED)";
            repBlurScore.textContent = q.blur_score;
            repBrightness.textContent = q.brightness;
            repContrast.textContent = q.contrast;
            repFov.textContent = q.retinal_area_ratio;
            repAlerts.textContent = q.reasons && q.reasons.length > 0 ? q.reasons.join(", ") : "None";

            repIcdrGrade.textContent = "Ungradeable - Recapture Required";
            repSeverity.textContent = "Image Inadequate";
            repConfidence.textContent = "N/A (Halted)";
            repProbSummary.innerHTML = "<span style='color:#b45309; font-weight:600;'>Primary AI Screening Suspended: Image failed clinical quality gate</span>";
            repReferableScore.textContent = "N/A";
            repReferableStatus.textContent = "RECAPTURE REQUIRED";
            repRecommendation.textContent = data.recommendation;

            btnDownloadReportTxt.onclick = () => { window.location.href = data.outputs.report_download_url; };
            btnDownloadJson.onclick = () => { window.location.href = data.outputs.json_download_url; };
            return;
        }

        // Image Passed Quality Gate OR Research Override is Active
        if (qualityRejectionCard) qualityRejectionCard.style.display = "none";
        if (coreResultsGrid) coreResultsGrid.style.display = "grid";
        if (explainabilityStudioCard) explainabilityStudioCard.style.display = "block";

        if (data.is_research_override) {
            if (overrideWarningBanner) overrideWarningBanner.style.display = "block";
        } else {
            if (overrideWarningBanner) overrideWarningBanner.style.display = "none";
        }

        // Step 2: DR Severity
        const predClass = data.predicted_class;
        drLevelText.textContent = `Level ${predClass}`;
        drSeverityName.textContent = data.severity;
        icdrGradeText.textContent = data.icdr_grade;
        classConfidenceText.textContent = `${data.class_confidence_percent}%`;
        severityHeroBadge.className = `severity-badge-hero sev-${predClass}`;

        // 5 Class Probabilities
        probabilityBarsList.innerHTML = "";
        const classNames = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"];
        const probPercents = data.class_probabilities_percent || {};
        const barColors = ["#10b981", "#0ea5e9", "#f59e0b", "#f97316", "#ef4444"];

        classNames.forEach((name, idx) => {
            const pct = probPercents[name] !== undefined ? probPercents[name] : 0;
            const row = document.createElement("div");
            row.className = "prob-row";
            row.innerHTML = `
                <span class="prob-label">${name}</span>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${pct}%; background-color: ${barColors[idx]};"></div>
                </div>
                <span class="prob-value">${pct.toFixed(2)}%</span>
            `;
            probabilityBarsList.appendChild(row);
        });

        // Step 3: Referable DR Screening
        const isRef = data.referable;
        referableStatusText.textContent = isRef ? "REFERABLE" : "NON-REFERABLE";
        referableStatusDesc.textContent = data.recommendation;
        referableStatusHero.className = `referable-status-hero ${isRef ? "ref-yes" : "ref-no"}`;
        referableScoreVal.textContent = `${data.referable_score_percent}%`;

        const gaugeWidth = Math.min(Math.max(data.referable_score_percent, 0), 100);
        referableGaugeFill.style.width = `${gaugeWidth}%`;
        referralDetailText.textContent = data.referral_detail;

        // Step 4: Explainability Studio (Grad-CAM)
        if (data.outputs && data.outputs.enhanced_url) {
            enhancedResultImg.src = data.outputs.enhanced_url + `?t=${Date.now()}`;
            modalEnhancedImg.src = enhancedResultImg.src;
        }
        if (data.outputs && data.outputs.gradcam_url) {
            gradcamResultImg.src = data.outputs.gradcam_url + `?t=${Date.now()}`;
            modalGradcamImg.src = gradcamResultImg.src;
        }

        // Reset Lesions View
        lesionResultsView.style.display = "none";
        btnRunLesionAnalysis.disabled = false;
        btnRunLesionText.textContent = "Run Experimental Lesion Analysis";
        lesionSpinner.style.display = "none";

        // Step 5: Clinical Screening Report Paper
        reportDateText.textContent = `Date: ${data.timestamp}`;
        if (reportCaseIdText) reportCaseIdText.textContent = `Case ID: ${data.case_id || "NS-2026-09-18-001"}`;
        reportImageIdText.textContent = `Image: ${data.image}`;

        repQualityStatus.textContent = q.status + (data.is_research_override ? " (OVERRIDDEN)" : "");
        repBlurScore.textContent = q.blur_score;
        repBrightness.textContent = q.brightness;
        repContrast.textContent = q.contrast;
        repFov.textContent = q.retinal_area_ratio;
        repAlerts.textContent = q.reasons && q.reasons.length > 0 ? q.reasons.join(", ") : "None";

        repIcdrGrade.textContent = data.icdr_grade;
        repSeverity.textContent = data.severity;
        repConfidence.textContent = `${data.class_confidence_percent}%`;

        const probTokens = classNames.map((c) => `${c}: ${probPercents[c] || 0}%`);
        repProbSummary.innerHTML = probTokens.map((t) => `<span>${t}</span>`).join(" • ");

        repReferableScore.textContent = `${data.referable_score_percent}%`;
        repReferableStatus.textContent = isRef ? "YES" : "NO";
        repRecommendation.textContent = data.recommendation;

        btnDownloadReportTxt.onclick = () => { window.location.href = data.outputs.report_download_url; };
        btnDownloadJson.onclick = () => { window.location.href = data.outputs.json_download_url; };
    }

    // Explainability Controls
    let isSideBySide = true;
    btnToggleSideBySide.addEventListener("click", () => {
        isSideBySide = !isSideBySide;
        comparisonGrid.style.gridTemplateColumns = isSideBySide ? "1fr 1fr" : "1fr";
        btnToggleSideBySide.textContent = isSideBySide ? "Side-by-Side" : "Stacked View";
    });

    btnViewLargerModal.addEventListener("click", () => { imageModal.style.display = "flex"; });
    btnCloseModal.addEventListener("click", () => { imageModal.style.display = "none"; });
    modalBackdrop.addEventListener("click", () => { imageModal.style.display = "none"; });

    // Accordion Toggle
    let isAccordionOpen = false;
    lesionAccordionToggle.addEventListener("click", () => {
        isAccordionOpen = !isAccordionOpen;
        lesionAccordionBody.style.display = isAccordionOpen ? "block" : "none";
        accordionArrow.style.transform = isAccordionOpen ? "rotate(180deg)" : "rotate(0deg)";
    });

    // Experimental Lesions On-Demand
    btnRunLesionAnalysis.addEventListener("click", async () => {
        if (!currentResultData || !currentResultData.base_name) {
            alert("Please analyze a retinal image first.");
            return;
        }

        btnRunLesionAnalysis.disabled = true;
        lesionSpinner.style.display = "inline-block";
        btnRunLesionText.textContent = "Running U-Net V2 Segmentation...";

        try {
            const resp = await fetch("/api/experimental-lesions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ base_name: currentResultData.base_name })
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.error || errData.message || "Lesion segmentation failed.");
            }

            const data = await resp.json();
            lesionsOverlayImg.src = data.lesions_overlay_url + `?t=${Date.now()}`;

            const s = data.lesion_stats;
            if (s) {
                lesionStatMA.textContent = `Coverage: ${s.Microaneurysm.area_coverage_percent}% (Detected: ${s.Microaneurysm.detected ? "Yes" : "No"})`;
                lesionStatHEM.textContent = `Coverage: ${s.Hemorrhage.area_coverage_percent}% (Detected: ${s.Hemorrhage.detected ? "Yes" : "No"})`;
                lesionStatEX.textContent = `Coverage: ${s["Hard Exudate"] ? s["Hard Exudate"].area_coverage_percent : 0}%`;
                lesionStatSE.textContent = `Coverage: ${s["Soft Exudate"] ? s["Soft Exudate"].area_coverage_percent : 0}%`;
            }

            lesionResultsView.style.display = "block";
            btnRunLesionText.textContent = "Re-run Lesion Analysis";
        } catch (err) {
            console.error("Lesion segmentation error:", err);
            alert("Lesion Analysis Error: " + err.message);
            btnRunLesionText.textContent = "Run Experimental Lesion Analysis";
        } finally {
            btnRunLesionAnalysis.disabled = false;
            lesionSpinner.style.display = "none";
        }
    });

    btnPrintReport.addEventListener("click", () => { window.print(); });

    // ============================================================
    // 7. REPORTS SUB-NAV & DYNAMIC DASHBOARDS
    // ============================================================

    btnSubnavHistory.addEventListener("click", () => {
        btnSubnavHistory.classList.add("active");
        btnSubnavMonthly.classList.remove("active");
        btnSubnavYearly.classList.remove("active");
        subviewHistory.style.display = "block";
        subviewMonthly.style.display = "none";
        subviewYearly.style.display = "none";
        loadScreeningHistory();
    });

    btnSubnavMonthly.addEventListener("click", () => {
        btnSubnavMonthly.classList.add("active");
        btnSubnavHistory.classList.remove("active");
        btnSubnavYearly.classList.remove("active");
        subviewMonthly.style.display = "block";
        subviewHistory.style.display = "none";
        subviewYearly.style.display = "none";
        loadMonthlyReport();
    });

    btnSubnavYearly.addEventListener("click", () => {
        btnSubnavYearly.classList.add("active");
        btnSubnavHistory.classList.remove("active");
        btnSubnavMonthly.classList.remove("active");
        subviewYearly.style.display = "block";
        subviewHistory.style.display = "none";
        subviewMonthly.style.display = "none";
        loadYearlyReport();
    });

    async function loadScreeningHistory(filter = "all", search = "") {
        try {
            historyTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:20px; color:#64748b;">Loading screening history...</td></tr>`;
            const resp = await fetch(`/api/history?filter=${filter}&search=${encodeURIComponent(search)}&limit=100`);
            if (!resp.ok) throw new Error("Failed to fetch history");
            const data = await resp.json();

            historyTableBody.innerHTML = "";
            if (data.screenings.length === 0) {
                const isLive = (data.data_mode || appMode) === "live";
                const emptyMsg = isLive
                    ? "No live screening records yet. Run your first screening to populate this table."
                    : "No matching screening records found.";
                historyTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:24px; color:#64748b;">${emptyMsg}</td></tr>`;
                return;
            }

            data.screenings.forEach((r) => {
                const tr = document.createElement("tr");

                let statusClass = "status-completed";
                if (r.processing_status === "REQUIRES SPECIALIST REVIEW") statusClass = "status-review-req";
                else if (r.processing_status === "NEEDS RECAPTURE") statusClass = "status-recapture";

                let revClass = "rev-none";
                if (r.review_status === "Pending Review") revClass = "rev-pending";
                else if (r.review_status === "Reviewed") revClass = "rev-done";
                else if (r.review_status === "Needs Further Assessment") revClass = "rev-further";

                // Source badge
                const srcColors = { upload: "#0284c7", camera: "#0d9488", sample: "#7c3aed", demo: "#f59e0b" };
                const srcColor = srcColors[r.source_type] || "#64748b";
                let srcBadgeText = (r.source_type || "UNKNOWN").toUpperCase();
                let srcTitle = "Screening source";

                if (r.source_type === "upload") {
                    srcBadgeText = "UPLOAD";
                    srcTitle = "Live/user image upload";
                } else if (r.source_type === "camera") {
                    srcBadgeText = "CAMERA";
                    srcTitle = "Live camera acquisition";
                } else if (r.source_type === "sample") {
                    srcBadgeText = "SAMPLE / TEST";
                    srcTitle = "Test/demo sample";
                } else if (r.source_type === "demo") {
                    srcBadgeText = "DEMO DATA";
                    srcTitle = "Synthetic simulated record";
                }

                const srcLabel = `<span class="badge" title="${srcTitle}" style="background:${srcColor};color:#fff;font-size:9px;font-weight:700;padding:2px 6px;border-radius:3px;letter-spacing:0.02em;">${srcBadgeText}</span>`;

                const conf = r.class_confidence > 0 ? (r.class_confidence * 100).toFixed(1) + "%" : "N/A";
                const refScore = r.referable_score > 0 ? (r.referable_score * 100).toFixed(1) + "%" : "N/A";

                tr.innerHTML = `
                    <td><strong>${r.case_id}</strong></td>
                    <td>${r.timestamp}</td>
                    <td>${srcLabel}</td>
                    <td><span class="badge ${r.quality_status === 'GOOD' ? 'badge-mint' : 'badge-amber'}">${r.quality_status}</span></td>
                    <td><strong>${r.icdr_grade || ('Level ' + r.predicted_class)}</strong></td>
                    <td>${conf}</td>
                    <td><strong>${refScore}</strong></td>
                    <td><span class="status-pill ${statusClass}">${r.processing_status}</span></td>
                    <td><span class="review-badge ${revClass}">${r.review_status}</span></td>
                `;
                historyTableBody.appendChild(tr);
            });
        } catch (e) {
            console.error("History load error:", e);
            historyTableBody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:#ef4444; padding:16px;">Failed to load screening history.</td></tr>`;
        }
    }

    filterPills.forEach((p) => {
        p.addEventListener("click", () => {
            filterPills.forEach((x) => x.classList.remove("active"));
            p.classList.add("active");
            loadScreeningHistory(p.getAttribute("data-filter"), historySearchInput.value);
        });
    });

    historySearchInput.addEventListener("input", () => {
        const activeFilter = document.querySelector(".filter-pill[data-filter].active")?.getAttribute("data-filter") || "all";
        loadScreeningHistory(activeFilter, historySearchInput.value);
    });

    btnExportCsv.addEventListener("click", () => {
        window.location.href = "/api/export/csv";
    });

    async function loadMonthlyReport() {
        const m = monthSelect ? monthSelect.value : 9;
        try {
            const resp = await fetch(`/api/reports/monthly?year=2026&month=${m}`);
            if (!resp.ok) throw new Error("Failed to fetch monthly report");
            const data = await resp.json();

            document.getElementById("monthTotal").textContent = data.total_screenings;
            document.getElementById("monthCompleted").textContent = data.completed;
            document.getElementById("monthRecapture").textContent = data.needs_recapture;
            document.getElementById("monthReview").textContent = data.specialist_review;

            document.getElementById("monthGoodQual").textContent = data.quality_summary.good;
            document.getElementById("monthPoorQual").textContent = data.quality_summary.review;
            document.getElementById("monthReferable").textContent = data.referable;
            document.getElementById("monthNonRef").textContent = data.non_referable;
            document.getElementById("monthAvgConf").textContent = `${data.avg_confidence}%`;
            document.getElementById("monthAvgTime").textContent = `${data.avg_processing_time_ms} ms`;
            document.getElementById("monthQualFailRate").textContent = `${data.quality_failure_rate}%`;
            document.getElementById("monthRefRate").textContent = `${data.referral_rate}%`;

            const monthDrDistList = document.getElementById("monthDrDistList");
            monthDrDistList.innerHTML = "";
            const labels = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"];
            labels.forEach((lvl) => {
                const count = data.dr_distribution[lvl] || 0;
                const pct = data.total_screenings > 0 ? (count / data.total_screenings * 100).toFixed(1) : 0;
                const row = document.createElement("div");
                row.className = "prob-row";
                row.innerHTML = `
                    <span class="prob-label">${lvl}</span>
                    <div class="prob-track"><div class="prob-fill" style="width:${pct}%; background:#0284c7;"></div></div>
                    <span class="prob-value">${count} (${pct}%)</span>
                `;
                monthDrDistList.appendChild(row);
            });
        } catch (e) {
            console.error("Monthly report error:", e);
        }
    }

    if (monthSelect) {
        monthSelect.addEventListener("change", loadMonthlyReport);
    }

    async function loadYearlyReport() {
        try {
            const resp = await fetch("/api/reports/yearly?year=2026");
            if (!resp.ok) throw new Error("Failed to fetch yearly report");
            const data = await resp.json();

            document.getElementById("yearTotal").textContent = data.total_screenings.toLocaleString();
            document.getElementById("yearCompleted").textContent = data.completed.toLocaleString();
            document.getElementById("yearRecapture").textContent = data.needs_recapture.toLocaleString();
            document.getElementById("yearReview").textContent = data.specialist_review.toLocaleString();

            const container = document.getElementById("yearBarsContainer");
            container.innerHTML = "";

            const maxMonth = Math.max(...data.monthly_breakdown.map((m) => m.total), 1);

            data.monthly_breakdown.forEach((mb) => {
                const heightPct = (mb.total / maxMonth * 100).toFixed(0);
                const col = document.createElement("div");
                col.className = "year-bar-col";
                col.innerHTML = `
                    <div class="year-bar-track">
                        <span class="year-bar-val">${mb.total > 0 ? mb.total : ''}</span>
                        <div class="year-bar-fill" style="height: ${heightPct}%;"></div>
                    </div>
                    <span class="year-bar-lbl">${mb.month_abbr}</span>
                `;
                container.appendChild(col);
            });
        } catch (e) {
            console.error("Yearly report error:", e);
        }
    }

    // ============================================================
    // 8. SPECIALIST REVIEW QUEUE
    // ============================================================

    async function loadReviewQueue(status = "all") {
        try {
            reviewTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:20px; color:#64748b;">Loading review queue...</td></tr>`;
            const resp = await fetch(`/api/review-queue?status=${encodeURIComponent(status)}`);
            if (!resp.ok) throw new Error("Failed to load review queue");
            const data = await resp.json();

            reviewTableBody.innerHTML = "";
            if (data.cases.length === 0) {
                reviewTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:24px; color:#64748b;">No cases in this review status.</td></tr>`;
                return;
            }

            data.cases.forEach((c) => {
                const tr = document.createElement("tr");

                let revClass = "rev-pending";
                if (c.review_status === "Reviewed") revClass = "rev-done";
                else if (c.review_status === "Needs Further Assessment") revClass = "rev-further";

                tr.innerHTML = `
                    <td><strong>${c.case_id}</strong></td>
                    <td>${c.timestamp}</td>
                    <td><span class="badge badge-amber">${c.icdr_grade}</span></td>
                    <td><strong>${(c.referable_score * 100).toFixed(1)}%</strong></td>
                    <td>${(c.class_confidence * 100).toFixed(1)}%</td>
                    <td><span class="review-badge ${revClass}">${c.review_status}</span></td>
                    <td>
                        <button type="button" class="btn btn-secondary btn-sm btn-open-review" data-case-id="${c.case_id}" data-grade="${c.icdr_grade}" data-score="${(c.referable_score * 100).toFixed(1)}%" data-status="${c.review_status}" data-notes="${c.review_notes || ''}">
                            Review Case
                        </button>
                    </td>
                `;
                reviewTableBody.appendChild(tr);
            });

            // Bind review case buttons
            document.querySelectorAll(".btn-open-review").forEach((btn) => {
                btn.addEventListener("click", () => {
                    currentReviewCaseId = btn.getAttribute("data-case-id");
                    revModalCaseId.textContent = currentReviewCaseId;
                    revModalGrade.textContent = btn.getAttribute("data-grade");
                    revModalScore.textContent = btn.getAttribute("data-score");
                    revModalStatusSelect.value = btn.getAttribute("data-status") || "Reviewed";
                    revModalNotes.value = btn.getAttribute("data-notes") || "";
                    reviewModal.style.display = "flex";
                });
            });
        } catch (e) {
            console.error("Review queue error:", e);
            reviewTableBody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:#ef4444; padding:16px;">Failed to load specialist review queue.</td></tr>`;
        }
    }

    reviewFilterPills.forEach((p) => {
        p.addEventListener("click", () => {
            reviewFilterPills.forEach((x) => x.classList.remove("active"));
            p.classList.add("active");
            loadReviewQueue(p.getAttribute("data-review-filter"));
        });
    });

    btnCloseReviewModal.addEventListener("click", () => { reviewModal.style.display = "none"; });
    btnCancelReviewModal.addEventListener("click", () => { reviewModal.style.display = "none"; });
    reviewModalBackdrop.addEventListener("click", () => { reviewModal.style.display = "none"; });

    btnSaveReviewModal.addEventListener("click", async () => {
        if (!currentReviewCaseId) return;

        btnSaveReviewModal.disabled = true;
        try {
            const resp = await fetch(`/api/review/${currentReviewCaseId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    status: revModalStatusSelect.value,
                    notes: revModalNotes.value
                })
            });

            if (!resp.ok) throw new Error("Failed to save review status");

            reviewModal.style.display = "none";
            loadReviewQueue();
            updateReviewQueueCount();
        } catch (e) {
            alert("Error updating review status: " + e.message);
        } finally {
            btnSaveReviewModal.disabled = false;
        }
    });

    // ============================================================
    // 9. ANALYTICS & 100k CAPACITY PLANNER
    // ============================================================

    async function loadAnalytics() {
        try {
            const resp = await fetch("/api/analytics");
            if (!resp.ok) throw new Error("Failed to fetch analytics");
            const data = await resp.json();

            // Primary LIVE KPIs (upload+camera only — never includes sample or demo)
            document.getElementById("anTotalCount").textContent = data.total_screenings.toLocaleString();
            document.getElementById("anAvgLatency").textContent =
                data.avg_processing_time_ms > 0 ? `${data.avg_processing_time_ms} ms` : "— ms";
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
    }

    async function updateCapacityCalculations() {
        const target = inputTargetPatients.value || 100000;
        const days = inputWorkingDays.value || 300;
        const clinics = inputClinics.value || 10;
        const latency = inputLatency.value || 0.12;

        try {
            const resp = await fetch(`/api/capacity?target=${target}&working_days=${days}&clinics=${clinics}&inference_time_sec=${latency}`);
            if (!resp.ok) throw new Error("Capacity query failed");
            const data = await resp.json();

            const tm = data.throughput_metrics;
            calcReqDay.textContent = Math.round(tm.required_screenings_per_day).toLocaleString();
            calcReqClinic.textContent = tm.required_per_clinic_day.toLocaleString();
            calcAiCapDay.textContent = tm.ai_daily_capacity_per_workstation.toLocaleString();
            calcSpecReviews.textContent = Math.round(tm.estimated_specialist_reviews_day).toLocaleString();
        } catch (e) {
            console.warn("Capacity calculation error:", e);
        }
    }

    [inputTargetPatients, inputWorkingDays, inputClinics, inputLatency].forEach((inp) => {
        if (inp) inp.addEventListener("input", updateCapacityCalculations);
    });
});
