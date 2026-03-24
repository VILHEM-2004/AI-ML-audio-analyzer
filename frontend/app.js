/**
 * AI Audio Analyzer – Frontend Application
 * Handles file upload, API communication, and result rendering.
 */

(function () {
    "use strict";

    // ─── DOM Elements ───
    const dropZone = document.getElementById("drop-zone");
    const dropZoneContent = document.getElementById("drop-zone-content");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const clearFileBtn = document.getElementById("clear-file-btn");
    const analyzeBtn = document.getElementById("analyze-btn");

    const uploadSection = document.getElementById("upload-section");
    const loadingSection = document.getElementById("loading-section");
    const errorSection = document.getElementById("error-section");
    const resultsSection = document.getElementById("results-section");
    const loaderStatus = document.getElementById("loader-status");
    const loaderProgressBar = document.getElementById("loader-progress-bar");
    const errorMessage = document.getElementById("error-message");
    const retryBtn = document.getElementById("retry-btn");
    const newAnalysisBtn = document.getElementById("new-analysis-btn");
    const toggleFeaturesBtn = document.getElementById("toggle-features");
    const featuresBody = document.getElementById("features-body");

    // ─── State ───
    let selectedFile = null;
    let progressInterval = null;

    // ─── File Upload ───
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    clearFileBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearFile();
    });

    analyzeBtn.addEventListener("click", () => analyzeAudio());
    retryBtn.addEventListener("click", () => showSection("upload"));
    newAnalysisBtn.addEventListener("click", () => {
        clearFile();
        showSection("upload");
    });

    toggleFeaturesBtn.addEventListener("click", () => {
        featuresBody.classList.toggle("open");
        toggleFeaturesBtn.classList.toggle("active");
    });

    // ─── File Handling ───
    function handleFile(file) {
        const validExtensions = [".wav", ".mp3", ".flac"];
        const ext = "." + file.name.split(".").pop().toLowerCase();

        if (!validExtensions.includes(ext)) {
            showError(`Unsupported file format "${ext}". Please upload WAV, MP3, or FLAC files.`);
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.classList.remove("hidden");
        analyzeBtn.classList.remove("hidden");
        analyzeBtn.disabled = false;
    }

    function clearFile() {
        selectedFile = null;
        fileInput.value = "";
        fileInfo.classList.add("hidden");
        analyzeBtn.classList.add("hidden");
        analyzeBtn.disabled = true;
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / 1048576).toFixed(1) + " MB";
    }

    // ─── Section Management ───
    function showSection(name) {
        uploadSection.classList.toggle("hidden", name !== "upload");
        loadingSection.classList.toggle("hidden", name !== "loading");
        errorSection.classList.toggle("hidden", name !== "error");
        resultsSection.classList.toggle("hidden", name !== "results");

        if (name === "results" || name === "upload") {
            window.scrollTo({ top: 0, behavior: "smooth" });
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        showSection("error");
    }

    // ─── Progress Simulation ───
    function startProgress() {
        let progress = 0;
        const stages = [
            { at: 10, text: "Loading audio file..." },
            { at: 25, text: "Extracting audio features..." },
            { at: 45, text: "Detecting tempo & key..." },
            { at: 65, text: "Classifying genre..." },
            { at: 80, text: "Generating visualizations..." },
            { at: 95, text: "Preparing results..." },
        ];

        loaderProgressBar.style.width = "0%";

        progressInterval = setInterval(() => {
            if (progress < 95) {
                progress += Math.random() * 3 + 0.5;
                progress = Math.min(progress, 95);
                loaderProgressBar.style.width = progress + "%";

                for (const stage of stages) {
                    if (progress >= stage.at && loaderStatus.textContent !== stage.text) {
                        loaderStatus.textContent = stage.text;
                    }
                }
            }
        }, 200);
    }

    function stopProgress() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
        loaderProgressBar.style.width = "100%";
        loaderStatus.textContent = "Complete!";
    }

    // ─── API Call ───
    async function analyzeAudio() {
        if (!selectedFile) return;

        showSection("loading");
        startProgress();

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });

            stopProgress();

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (${response.status})`);
            }

            const data = await response.json();
            renderResults(data);
            showSection("results");
        } catch (err) {
            stopProgress();
            showError(err.message || "Failed to analyze the audio file. Please try again.");
        }
    }

    // ─── Render Results ───
    function renderResults(data) {
        // Stats
        renderStats(data);
        // Genre bars
        renderGenreBars(data.genre);
        // Visualizations
        renderVisualizations(data.visualizations);
        // Feature details
        renderFeatureDetails(data.features);
    }

    function renderStats(data) {
        // Tempo
        const tempo = data.tempo_summary || data.tempo || {};
        setStatValue("stat-tempo-value", tempo.bpm ? Math.round(tempo.bpm) : "—");
        setConfidence("stat-tempo-confidence", tempo.confidence);

        // Key
        const key = data.key || {};
        setStatValue("stat-key-value", key.key_name || "—");
        setConfidence("stat-key-confidence", key.confidence);

        // Energy
        const features = data.features || {};
        setStatValue("stat-energy-value", features.energy_level || "—");

        // Genre
        const genre = data.genre || {};
        const genreName = genre.predicted_genre || "—";
        setStatValue("stat-genre-value", capitalize(genreName));
        setConfidence("stat-genre-confidence", genre.confidence);

        // Duration
        const info = data.file_info || {};
        setStatValue("stat-duration-value", info.duration ? formatDuration(info.duration) : "—");

        // Analysis time
        setStatValue(
            "stat-analysis-time-value",
            data.analysis_time ? data.analysis_time + "s" : "—"
        );
    }

    function setStatValue(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    function setConfidence(id, confidence) {
        const el = document.getElementById(id);
        if (!el) return;

        if (confidence) {
            el.textContent = confidence + " confidence";
            el.className = "stat-confidence " + confidence.toLowerCase();
        } else {
            el.textContent = "";
            el.className = "stat-confidence";
        }
    }

    function renderGenreBars(genre) {
        const container = document.getElementById("genre-bars");
        container.innerHTML = "";

        if (!genre || genre.error || !genre.probabilities) {
            container.innerHTML = '<p style="color: var(--text-muted); font-size: 0.9rem;">Genre classification unavailable.</p>';
            return;
        }

        const probs = genre.probabilities;
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
        const maxProb = sorted[0]?.[1] || 1;

        sorted.forEach(([name, prob], index) => {
            const item = document.createElement("div");
            item.className = "genre-bar-item";

            const percent = (prob * 100).toFixed(1);
            const barWidth = (prob / maxProb) * 100;

            item.innerHTML = `
                <span class="genre-bar-label">${capitalize(name)}</span>
                <div class="genre-bar-track">
                    <div class="genre-bar-fill rank-${Math.min(index, 7)}" style="width: 0%"></div>
                </div>
                <span class="genre-bar-value">${percent}%</span>
            `;

            container.appendChild(item);

            // Animate bar width
            requestAnimationFrame(() => {
                setTimeout(() => {
                    item.querySelector(".genre-bar-fill").style.width = barWidth + "%";
                }, 50 + index * 80);
            });
        });
    }

    function renderVisualizations(viz) {
        if (!viz || viz.error) return;

        const vizKeys = ["waveform", "spectrogram", "chromagram", "spectral_features", "mfcc"];

        vizKeys.forEach((key) => {
            const img = document.getElementById("img-" + key);
            if (img && viz[key]) {
                img.src = "data:image/png;base64," + viz[key];
                img.alt = capitalize(key.replace(/_/g, " ")) + " visualization";
            }
        });
    }

    function renderFeatureDetails(features) {
        const body = document.getElementById("features-body");
        body.innerHTML = "";

        if (!features || features.error) {
            body.innerHTML = '<p style="color: var(--text-muted);">Feature details unavailable.</p>';
            return;
        }

        // Build feature table
        const table = document.createElement("table");
        table.className = "features-table";

        const thead = document.createElement("thead");
        thead.innerHTML = `<tr><th>Feature</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>`;
        table.appendChild(thead);

        const tbody = document.createElement("tbody");

        const featureKeys = [
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "zero_crossing_rate",
            "rms_energy",
        ];

        featureKeys.forEach((key) => {
            const f = features[key];
            if (f && typeof f === "object" && f.mean !== undefined) {
                const row = document.createElement("tr");
                row.innerHTML = `
                    <td>${formatFeatureName(key)}</td>
                    <td>${formatNum(f.mean)}</td>
                    <td>${formatNum(f.std)}</td>
                    <td>${formatNum(f.min)}</td>
                    <td>${formatNum(f.max)}</td>
                `;
                tbody.appendChild(row);
            }
        });

        table.appendChild(tbody);
        body.appendChild(table);

        // Chroma profile chart
        if (features.chroma_profile) {
            const chromaTitle = document.createElement("h4");
            chromaTitle.style.cssText = "color: var(--text-primary); margin: var(--space-lg) 0 var(--space-md); font-size: 0.9rem;";
            chromaTitle.textContent = "Pitch Class Distribution (Chroma)";
            body.appendChild(chromaTitle);

            const chart = document.createElement("div");
            chart.className = "chroma-chart";

            const chromaVals = Object.values(features.chroma_profile);
            const maxChroma = Math.max(...chromaVals, 0.01);

            const colors = [
                "#6c63ff", "#7c6fff", "#8c7bff", "#00d4ff", "#00c4e8",
                "#00b4d1", "#ff6b9d", "#ff7ba8", "#ff8bb3", "#ffa726",
                "#ffb74d", "#ffc107"
            ];

            Object.entries(features.chroma_profile).forEach(([note, val], i) => {
                const wrapper = document.createElement("div");
                wrapper.className = "chroma-bar-wrapper";

                const bar = document.createElement("div");
                bar.className = "chroma-bar";
                bar.style.height = "0%";
                bar.style.background = colors[i % colors.length];
                bar.style.marginTop = "auto";

                const label = document.createElement("span");
                label.className = "chroma-label";
                label.textContent = note;

                wrapper.appendChild(bar);
                wrapper.appendChild(label);
                chart.appendChild(wrapper);

                // Animate
                requestAnimationFrame(() => {
                    setTimeout(() => {
                        bar.style.height = ((val / maxChroma) * 100) + "%";
                    }, 100 + i * 50);
                });
            });

            body.appendChild(chart);
        }

        // Open features panel by default
        featuresBody.classList.add("open");
        toggleFeaturesBtn.classList.add("active");
    }

    // ─── Utilities ───
    function capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }

    function formatDuration(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, "0")}`;
    }

    function formatFeatureName(key) {
        return key
            .split("_")
            .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");
    }

    function formatNum(n) {
        if (typeof n !== "number") return "—";
        if (Math.abs(n) >= 1000) return n.toFixed(1);
        if (Math.abs(n) >= 1) return n.toFixed(3);
        return n.toFixed(6);
    }
})();
