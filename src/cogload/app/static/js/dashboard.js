// Cogload Precision Spatial Telemetry & 3D Interactive Console
document.addEventListener("DOMContentLoaded", () => {
  // 1. Primary Telemetry Elements
  const loadBadge = document.getElementById("load-badge");
  const confidenceVal = document.getElementById("confidence-val");
  const scoreVal = document.getElementById("score-val");

  const chipFace = document.getElementById("chip-face");
  const chipCalib = document.getElementById("chip-calib");
  const fpsBadge = document.getElementById("fps-badge");

  const indBlink = document.getElementById("ind-blink");
  const deltaBlink = document.getElementById("delta-blink");
  const indEar = document.getElementById("ind-ear");
  const indGaze = document.getElementById("ind-gaze");
  const deltaGaze = document.getElementById("delta-gaze");
  const indHead = document.getElementById("ind-head");
  const deltaHead = document.getElementById("delta-head");
  const indTension = document.getElementById("ind-tension");
  const deltaTension = document.getElementById("delta-tension");

  const warningsBox = document.getElementById("warnings-box");
  const warningText = document.getElementById("warning-text");

  const toggleOverlay = document.getElementById("toggle-overlay");
  const liveCanvas = document.getElementById("live-canvas");
  const canvasCtx = liveCanvas ? liveCanvas.getContext("2d") : null;
  const segments = document.querySelectorAll("#segment-grid .segment");

  // =========================================================================
  // 2. Hardware 3D Tilt & Specular Light Parallax Engine (Apple Vision-Grade)
  // =========================================================================
  const interactive3DElements = document.querySelectorAll(".card, .indicator-tile");

  interactive3DElements.forEach((el) => {
    let bounds;
    
    function updateBounds() {
      bounds = el.getBoundingClientRect();
    }

    el.addEventListener("mouseenter", updateBounds);

    el.addEventListener("mousemove", (e) => {
      if (!bounds) updateBounds();
      const mouseX = e.clientX - bounds.left;
      const mouseY = e.clientY - bounds.top;
      
      const xPct = mouseX / bounds.width;
      const yPct = mouseY / bounds.height;
      
      // Specular light sheen coordinate
      el.style.setProperty("--mouse-x", `${mouseX}px`);
      el.style.setProperty("--mouse-y", `${mouseY}px`);
      el.style.setProperty("--tile-x", `${mouseX}px`);
      el.style.setProperty("--tile-y", `${mouseY}px`);

      // Gentle, physical 3D rotational tilt (-3.5deg to +3.5deg)
      const rotX = ((0.5 - yPct) * 6.0).toFixed(2);
      const rotY = ((xPct - 0.5) * 6.0).toFixed(2);

      el.style.transform = `perspective(1000px) rotateX(${rotX}deg) rotateY(${rotY}deg) scale3d(1.008, 1.008, 1.008)`;
    });

    el.addEventListener("mouseleave", () => {
      el.style.transform = "perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)";
    });
  });

  // =========================================================================
  // 3. 20-Segment Physical Tactile LED Meter Controller
  // =========================================================================
  function updateSegmentMeter(score) {
    if (!segments || segments.length === 0) return;
    const clampedScore = Math.max(0, Math.min(1.0, score));
    const activeCount = Math.round(clampedScore * segments.length);

    segments.forEach((seg, idx) => {
      seg.className = "segment";
      if (idx < activeCount) {
        if (idx < 8) {
          seg.classList.add("lit-low");
        } else if (idx < 15) {
          seg.classList.add("lit-med");
        } else {
          seg.classList.add("lit-high");
        }
      }
    });
  }

  // =========================================================================
  // 4. Universal High-Performance 30 FPS Canvas Viewfinder
  // =========================================================================
  if (liveCanvas && canvasCtx) {
    let frameActive = true;
    function fetchAndRenderFrame() {
      if (!frameActive) return;
      const overlayOn = toggleOverlay ? toggleOverlay.checked : true;
      const img = new Image();
      img.onload = () => {
        canvasCtx.drawImage(img, 0, 0, liveCanvas.width, liveCanvas.height);
        requestAnimationFrame(fetchAndRenderFrame);
      };
      img.onerror = () => {
        setTimeout(fetchAndRenderFrame, 80);
      };
      img.src = `/api/frame?overlay=${overlayOn}&_t=${Date.now()}`;
    }
    fetchAndRenderFrame();
  }

  // =========================================================================
  // 5. Studio-Calibrated Darkroom Chrono-Telemetry Chart
  // =========================================================================
  const chartCanvas = document.getElementById("timelineChart");
  const ctx = chartCanvas.getContext("2d");
  const maxDataPoints = 60; // 60-second rolling window

  // Area vertical gradient fill
  const areaGradient = ctx.createLinearGradient(0, 0, 0, 220);
  areaGradient.addColorStop(0, "rgba(14, 165, 233, 0.35)");
  areaGradient.addColorStop(0.65, "rgba(14, 165, 233, 0.08)");
  areaGradient.addColorStop(1, "rgba(14, 165, 233, 0.0)");

  const chartData = {
    labels: Array(maxDataPoints).fill(""),
    datasets: [{
      label: "Cognitive Load",
      data: Array(maxDataPoints).fill(0),
      borderColor: "#38BDF8",
      backgroundColor: areaGradient,
      borderWidth: 2.2,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: "#FFFFFF",
      pointHoverBorderColor: "#38BDF8",
      fill: true,
      tension: 0.35,
    }]
  };

  const timelineChart = new Chart(ctx, {
    type: "line",
    data: chartData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        y: {
          min: 0,
          max: 1.0,
          grid: {
            color: "rgba(255, 255, 255, 0.04)",
            drawBorder: false,
          },
          ticks: {
            color: "#64748B",
            font: {
              family: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif",
              size: 10,
              weight: "600",
            },
            stepSize: 0.25,
            callback: (val) => val.toFixed(2),
          }
        },
        x: {
          grid: { display: false },
          ticks: { display: false }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(12, 15, 23, 0.95)",
          titleColor: "#94A3B8",
          bodyColor: "#F0F4F8",
          borderColor: "rgba(255, 255, 255, 0.12)",
          borderWidth: 1,
          padding: 8,
          displayColors: false,
          callbacks: {
            label: (item) => `Workload Index: ${Number(item.raw).toFixed(3)}`
          }
        }
      }
    }
  });

  // =========================================================================
  // 6. Real-Time Telemetry Stream Processor (SSE)
  // =========================================================================
  const eventSource = new EventSource("/api/stream");

  eventSource.onmessage = (event) => {
    try {
      const sanitized = event.data.replace(/:\s*NaN\b/g, ': 0.0').replace(/:\s*Infinity\b/g, ': 999.0');
      const data = JSON.parse(sanitized);

      // Status Micro-Chips
      if (data.face_detected) {
        chipFace.textContent = "Face Locked";
        chipFace.className = "chip chip-success";
      } else {
        chipFace.textContent = "Searching Face";
        chipFace.className = "chip chip-warning";
      }

      if (data.calibrated) {
        chipCalib.textContent = "Calibrated";
        chipCalib.className = "chip chip-success";
      } else {
        chipCalib.textContent = "Uncalibrated Baseline";
        chipCalib.className = "chip chip-muted";
      }

      fpsBadge.textContent = `${data.fps} FPS`;

      // Primary Workload State Badge
      const level = data.load.level;
      loadBadge.textContent = level;
      loadBadge.className = "load-badge " + (
        level === "HIGH" ? "load-high" : level === "MEDIUM" ? "load-med" : "load-low"
      );

      const confPct = Math.round((data.load.confidence || 0) * 100);
      confidenceVal.textContent = `${confPct}%`;
      const score = data.load.score || 0.0;
      scoreVal.textContent = score.toFixed(2);

      // Update 20-Segment Physical Tactile LED Meter
      updateSegmentMeter(score);

      // Indicator Readouts & Delta Baseline Pills
      if (data.indicators) {
        indBlink.textContent = data.indicators.blink_rate;
        indEar.textContent = data.indicators.ear;
        indGaze.textContent = data.indicators.gaze_dispersion;
        indHead.textContent = data.indicators.head_motion;
        if (indTension) indTension.textContent = data.indicators.brow_furrow || "--";

        if (data.calibrated && data.indicators.z) {
          const zb = data.indicators.z.blink_rate;
          const zg = data.indicators.z.gaze_dispersion;
          const zh = data.indicators.z.head_motion;
          const zt = data.indicators.z.brow_furrow;
          deltaBlink.textContent = `Δ ${zb > 0 ? "+" : ""}${zb} σ`;
          deltaGaze.textContent = `Δ ${zg > 0 ? "+" : ""}${zg} σ`;
          deltaHead.textContent = `Δ ${zh > 0 ? "+" : ""}${zh} σ`;
          if (deltaTension) deltaTension.textContent = `Δ ${zt > 0 ? "+" : ""}${zt} σ`;
        } else {
          deltaBlink.textContent = "Δ Uncalibrated";
          deltaGaze.textContent = "Δ Uncalibrated";
          deltaHead.textContent = "Δ Uncalibrated";
          if (deltaTension) deltaTension.textContent = "Δ Uncalibrated";
        }
      }

      // Dynamic Annunciator Warnings (Oral Activity / Head Motion)
      if (data.warnings && data.warnings.length > 0) {
        const readable = data.warnings.map(w => {
          if (w === "not_calibrated") return "Run calibration for personalized baseline deviation";
          if (w === "no_face") return "No face detected in camera viewport";
          if (w === "high_head_motion") return "Excessive head angular velocity detected";
          if (w === "oral_activity") return "Oral activity (eating / speaking) — jaw artifacts actively filtered";
          if (w === "model_missing") return "Indicators-only mode (inference model missing)";
          return w;
        });
        warningText.textContent = readable.join(" · ");
        warningsBox.classList.remove("hidden");
      } else {
        warningsBox.classList.add("hidden");
      }

      // Push to Chrono-Telemetry Chart
      chartData.datasets[0].data.push(score);
      chartData.datasets[0].data.shift();
      timelineChart.update();

    } catch (err) {
      console.error("Error processing telemetry packet:", err);
    }
  };

  eventSource.onerror = (err) => {
    console.warn("SSE telemetry stream interrupted. Reconnecting...", err);
  };
});
