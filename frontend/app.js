/**
 * Application: frontend/app.js
 * Description: Client-side logic for StudentSignals early-warning system.
 *              Integrates with FastAPI REST backend endpoints.
 */

// Dynamic API Base URL resolution (supports Localhost, Render, and Vercel)
const DEFAULT_RENDER_API = "https://early-waring-student-performance.onrender.com";
const isLocalHost =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.protocol === "file:";

let API_BASE =
  localStorage.getItem("STUDENTSIGNALS_API_BASE") ||
  window.__API_URL__ ||
  (isLocalHost ? "" : DEFAULT_RENDER_API);

let currentMode = "early_warning";
let currentDataset = "mat";
let currentView = "dashboard";
let currentTier = "All";
let searchQuery = "";

let chartByType = null;
let chartByTier = null;
let chartLocalShap = null;
let chartGlobalShap = null;
let chartElements = null;
let chartSimProb = null;

let allStudents = [];

// ==============================================================================
// INITIALIZATION
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
  lucide.createIcons();
  updateApiStatusBadge();
  loadOverview();
  loadStudents();
  loadInsights();
  loadBenchmarks();
});

function updateApiStatusBadge() {
  const badgeText = document.getElementById("api-status-text");
  const dot = document.getElementById("api-status-dot");
  if (!badgeText || !dot) return;

  if (API_BASE) {
    try {
      const u = new URL(API_BASE);
      badgeText.innerText = `API: ${u.hostname.replace('.onrender.com', '')}`;
    } catch (e) {
      badgeText.innerText = "API: Custom Render";
    }
    dot.className = "w-2 h-2 rounded-full bg-emerald-500 animate-pulse";
  } else if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
    badgeText.innerText = "API: Localhost (8000)";
    dot.className = "w-2 h-2 rounded-full bg-emerald-500";
  } else {
    badgeText.innerText = "Connect Render API";
    dot.className = "w-2 h-2 rounded-full bg-amber-500 animate-bounce";
  }
}

function openApiConfigModal() {
  const current = localStorage.getItem("STUDENTSIGNALS_API_BASE") || DEFAULT_RENDER_API;
  const input = prompt(
    "Configure Backend API URL for Render & Vercel deployment:\n\n" +
    `• Default Render Backend: ${DEFAULT_RENDER_API}\n` +
    "• Leave empty to reset to default",
    current
  );

  if (input !== null) {
    const cleanUrl = input.trim().replace(/\/$/, "");
    if (cleanUrl) {
      localStorage.setItem("STUDENTSIGNALS_API_BASE", cleanUrl);
      API_BASE = cleanUrl;
    } else {
      localStorage.removeItem("STUDENTSIGNALS_API_BASE");
      API_BASE = isLocalHost ? "" : DEFAULT_RENDER_API;
    }
    updateApiStatusBadge();
    loadOverview();
    loadStudents();
    loadInsights();
    loadBenchmarks();
  }
}

// ==============================================================================
// NAVIGATION & VIEW SWITCHING
// ==============================================================================

function switchView(viewName) {
  currentView = viewName;

  const views = [
    "dashboard",
    "upload",
    "risk-list",
    "interventions",
    "insights",
    "benchmarks",
    "whatif",
  ];

  views.forEach((v) => {
    const el = document.getElementById(`view-${v}`);
    if (el) el.classList.add("hidden");

    const navBtn = document.getElementById(`nav-${v}`);
    if (navBtn) {
      navBtn.classList.remove("active", "bg-slate-800", "text-white");
      navBtn.classList.add("text-slate-400");
    }
  });

  const activeEl = document.getElementById(`view-${viewName}`);
  if (activeEl) activeEl.classList.remove("hidden");

  const activeNav = document.getElementById(`nav-${viewName}`);
  if (activeNav) {
    activeNav.classList.add("active", "text-white");
    activeNav.classList.remove("text-slate-400");
  }

  // Refresh data corresponding to view
  if (viewName === "dashboard") loadOverview();
  if (viewName === "risk-list") loadStudents();
  if (viewName === "interventions") loadInterventions();
  if (viewName === "insights") loadInsights();
  if (viewName === "benchmarks") loadBenchmarks();
  if (viewName === "whatif") populateSimulatorDropdown();

  lucide.createIcons();
}

function setViewingMode(mode) {
  currentMode = mode;
  const pillEarly = document.getElementById("pill-early");
  const pillFull = document.getElementById("pill-full");
  const label = document.getElementById("current-mode-label");

  if (mode === "early_warning") {
    pillEarly.className = "flex-1 py-1.5 px-2 rounded-md text-xs font-semibold bg-blue-600 text-white transition";
    pillFull.className = "flex-1 py-1.5 px-2 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition";
    label.innerText = "Demo Mode • Early-Warning (No Downstream Marks)";
  } else {
    pillFull.className = "flex-1 py-1.5 px-2 rounded-md text-xs font-semibold bg-blue-600 text-white transition";
    pillEarly.className = "flex-1 py-1.5 px-2 rounded-md text-xs font-semibold text-slate-400 hover:text-white transition";
    label.innerText = "Benchmark Mode • Full-Feature (Includes G2)";
  }

  loadOverview();
  loadStudents();
  loadInsights();
  loadBenchmarks();
}

function changeDataset(ds) {
  currentDataset = ds;
  const dsLabel = document.getElementById("current-dataset-label");
  dsLabel.innerText = ds === "mat" ? "Dataset: Mathematics (395 students across 2 schools)" : "Dataset: Portuguese (649 students across 2 schools)";

  loadOverview();
  loadStudents();
  loadInsights();
  loadBenchmarks();
}

// ==============================================================================
// VIEW 1: DASHBOARD OVERVIEW & CHARTS
// ==============================================================================

async function loadOverview() {
  try {
    const res = await fetch(`${API_BASE}/api/overview?mode=${currentMode}&dataset=${currentDataset}`);
    const data = await res.json();

    // Populate 5 KPI Cards
    document.getElementById("kpi-total").innerText = data.kpis.total_students;
    document.getElementById("kpi-high-risk").innerText = data.kpis.high_risk;
    document.getElementById("kpi-attendance").innerText = data.kpis.attendance_warnings;
    document.getElementById("kpi-grade").innerText = data.kpis.grade_warnings;
    document.getElementById("kpi-discipline").innerText = data.kpis.discipline_incidents;

    // Populate 4 Stat Pills
    document.getElementById("int-active").innerText = data.intervention_summary.active;
    document.getElementById("int-completed").innerText = data.intervention_summary.completed;
    document.getElementById("int-served").innerText = data.intervention_summary.students_served;
    document.getElementById("int-total").innerText = data.intervention_summary.total_interventions;

    // Render "By Type" Horizontal Bar Chart (ApexCharts)
    renderByTypeChart(data.by_type);

    // Render "By Tier" Donut Chart (ApexCharts)
    renderByTierChart(data.by_tier);

    lucide.createIcons();
  } catch (err) {
    console.error("Failed to load dashboard overview:", err);
  }
}

function renderByTypeChart(byTypeData) {
  const chartEl = document.querySelector("#chart-by-type");
  if (!chartEl) return;

  const categories = byTypeData.map((d) => d.type);
  const seriesData = byTypeData.map((d) => d.count);

  const options = {
    chart: {
      type: "bar",
      height: 240,
      toolbar: { show: false },
      fontFamily: "Plus Jakarta Sans, sans-serif",
    },
    plotOptions: {
      bar: {
        horizontal: true,
        borderRadius: 4,
        barHeight: "55%",
      },
    },
    colors: ["#0284c7"],
    dataLabels: { enabled: false },
    series: [{ name: "Active Cases", data: seriesData }],
    xaxis: {
      categories: categories,
      labels: { style: { colors: "#64748b", fontSize: "11px" } },
      axisBorder: { show: false },
    },
    yaxis: {
      labels: { style: { colors: "#334155", fontSize: "11px", fontWeight: 500 } },
    },
    grid: {
      borderColor: "#f1f5f9",
      strokeDashArray: 3,
    },
  };

  if (chartByType) {
    chartByType.destroy();
  }
  chartByType = new ApexCharts(chartEl, options);
  chartByType.render();
}

function renderByTierChart(byTierData) {
  const chartEl = document.querySelector("#chart-by-tier");
  if (!chartEl) return;

  const labels = byTierData.map((d) => d.tier);
  const series = byTierData.map((d) => d.count);
  const colors = byTierData.map((d) => d.color);

  const options = {
    chart: {
      type: "donut",
      height: 240,
      fontFamily: "Plus Jakarta Sans, sans-serif",
    },
    labels: labels,
    series: series,
    colors: colors,
    legend: {
      position: "bottom",
      fontSize: "11px",
      fontWeight: 500,
      labels: { colors: "#64748b" },
    },
    plotOptions: {
      pie: {
        donut: {
          size: "65%",
          labels: {
            show: true,
            total: {
              show: true,
              label: "Total Students",
              color: "#64748b",
              fontSize: "11px",
            },
          },
        },
      },
    },
    dataLabels: { enabled: false },
  };

  if (chartByTier) {
    chartByTier.destroy();
  }
  chartByTier = new ApexCharts(chartEl, options);
  chartByTier.render();
}

// ==============================================================================
// VIEW 2: STUDENT RISK LIST & DETAIL MODAL
// ==============================================================================

async function loadStudents() {
  try {
    let url = `${API_BASE}/api/students?mode=${currentMode}&dataset=${currentDataset}`;
    if (currentTier !== "All") url += `&tier=${encodeURIComponent(currentTier)}`;
    if (searchQuery) url += `&search=${encodeURIComponent(searchQuery)}`;

    const res = await fetch(url);
    allStudents = await res.json();

    const tbody = document.getElementById("students-table-body");
    tbody.innerHTML = "";

    if (allStudents.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" class="text-center py-8 text-slate-400">No students match the active filter criteria.</td></tr>`;
      return;
    }

    allStudents.forEach((s) => {
      let badgeClass = "badge-low";
      if (s.predicted_risk_tier === "High Risk") badgeClass = "badge-high";
      if (s.predicted_risk_tier === "Medium Risk") badgeClass = "badge-med";

      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-50/80 transition cursor-pointer";
      tr.onclick = () => openStudentModal(s.student_id);

      tr.innerHTML = `
        <td class="px-5 py-3 font-bold text-slate-900">${s.student_id}</td>
        <td class="px-4 py-3"><span class="${badgeClass}">${s.predicted_risk_tier}</span></td>
        <td class="px-4 py-3 text-slate-700 font-semibold">${s.risk_score || 0}%</td>
        <td class="px-4 py-3 font-bold text-slate-900">${(s.predicted_final_score || 0).toFixed(1)} / 20</td>
        <td class="px-4 py-3 text-slate-600">${s.G1 || "--"} / 20</td>
        <td class="px-4 py-3 text-slate-600">${s.studytime || "--"}</td>
        <td class="px-4 py-3 text-slate-600">${s.failures !== undefined ? s.failures : "--"}</td>
        <td class="px-4 py-3 text-slate-600">${s.absences !== undefined ? s.absences : "--"}</td>
        <td class="px-4 py-3 text-right">
          <button class="px-2.5 py-1 text-xs font-semibold text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-md transition">
            Inspect
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    populateSimulatorDropdown();
    lucide.createIcons();
  } catch (err) {
    console.error("Failed to load student risk list:", err);
  }
}

function filterRiskTier(tier) {
  currentTier = tier;
  const buttons = ["all", "high", "med", "low"];
  buttons.forEach((b) => {
    const el = document.getElementById(`btn-tier-${b}`);
    if (el) {
      el.className = "px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 transition";
    }
  });

  const activeId = tier === "All" ? "all" : tier === "High Risk" ? "high" : tier === "Medium Risk" ? "med" : "low";
  const activeBtn = document.getElementById(`btn-tier-${activeId}`);
  if (activeBtn) activeBtn.className = "px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white transition";

  loadStudents();
}

function handleSearch(val) {
  searchQuery = val.trim();
  loadStudents();
}

async function openStudentModal(studentId) {
  try {
    const res = await fetch(`${API_BASE}/api/student/${studentId}?mode=${currentMode}&dataset=${currentDataset}`);
    const data = await res.json();
    const p = data.profile;

    const modal = document.getElementById("student-modal");
    modal.classList.remove("hidden");

    document.getElementById("modal-student-id").innerText = `Student Diagnostic Inspector: ${p.student_id}`;
    document.getElementById("modal-student-sub").innerText = `School: ${p.school || "GP"} • Sex: ${p.sex || "F"} • Age: ${p.age || 17}`;

    // Quick metric cards
    let badgeClass = p.predicted_risk_tier === "High Risk" ? "badge-high" : p.predicted_risk_tier === "Medium Risk" ? "badge-med" : "badge-low";
    document.getElementById("modal-metrics").innerHTML = `
      <div class="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <div class="text-[11px] text-slate-500 font-medium">Risk Status</div>
        <div class="mt-1"><span class="${badgeClass}">${p.predicted_risk_tier}</span></div>
      </div>
      <div class="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <div class="text-[11px] text-slate-500 font-medium">Predicted Final Grade</div>
        <div class="text-base font-bold text-slate-900 mt-1">${(p.predicted_final_score || 0).toFixed(1)} / 20</div>
      </div>
      <div class="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <div class="text-[11px] text-slate-500 font-medium">Early Grade (G1)</div>
        <div class="text-base font-bold text-slate-900 mt-1">${p.G1 || "--"} / 20</div>
      </div>
      <div class="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <div class="text-[11px] text-slate-500 font-medium">Prior Failures</div>
        <div class="text-base font-bold text-slate-900 mt-1">${p.failures || 0}</div>
      </div>
      <div class="bg-slate-50 border border-slate-200 rounded-lg p-3">
        <div class="text-[11px] text-slate-500 font-medium">Absences</div>
        <div class="text-base font-bold text-slate-900 mt-1">${p.absences || 0} sessions</div>
      </div>
    `;

    // Render Local SHAP Waterfall
    renderLocalShapChart(data.local_attribution);

    // Render Recommendations
    const recBox = document.getElementById("modal-recommendations");
    recBox.innerHTML = "";
    data.recommendations.forEach((r) => {
      let badgeCol = r.priority === "Urgent" ? "bg-red-100 text-red-800" : r.priority === "High" ? "bg-amber-100 text-amber-800" : "bg-blue-100 text-blue-800";
      const item = document.createElement("div");
      item.className = "bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-1.5 text-xs";
      item.innerHTML = `
        <div class="flex items-center justify-between">
          <span class="font-bold text-slate-800">${r.title}</span>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold ${badgeCol}">${r.priority}</span>
        </div>
        <p class="text-slate-600 font-medium">${r.finding}</p>
        <ul class="list-disc pl-4 space-y-0.5 text-slate-500 text-[11px]">
          ${r.action_items.map((act) => `<li>${act}</li>`).join("")}
        </ul>
      `;
      recBox.appendChild(item);
    });

    modal.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error("Failed to load student details:", err);
  }
}

function closeStudentModal() {
  document.getElementById("student-modal").classList.add("hidden");
}

function renderLocalShapChart(attribution) {
  const chartEl = document.querySelector("#chart-local-shap");
  if (!chartEl) return;

  const drivers = (attribution.risk_drivers || []).slice(0, 4);
  const protect = (attribution.protective_factors || []).slice(0, 4);

  const categories = [];
  const seriesData = [];
  const colors = [];

  drivers.forEach((d) => {
    categories.push(d.feature.substring(0, 16));
    seriesData.push(parseFloat(d.attribution.toFixed(3)));
    colors.push("#ef4444");
  });

  protect.forEach((p) => {
    categories.push(p.feature.substring(0, 16));
    seriesData.push(parseFloat(p.attribution.toFixed(3)));
    colors.push("#10b981");
  });

  const options = {
    chart: {
      type: "bar",
      height: 240,
      toolbar: { show: false },
      fontFamily: "Plus Jakarta Sans, sans-serif",
    },
    plotOptions: {
      bar: {
        horizontal: true,
        borderRadius: 4,
        colors: {
          ranges: [
            { from: -100, to: 0, color: "#10b981" },
            { from: 0.001, to: 100, color: "#ef4444" },
          ],
        },
      },
    },
    series: [{ name: "SHAP Impact", data: seriesData }],
    xaxis: {
      categories: categories,
      labels: { style: { colors: "#64748b", fontSize: "10px" } },
    },
    yaxis: {
      labels: { style: { colors: "#334155", fontSize: "11px", fontWeight: 500 } },
    },
    grid: { borderColor: "#f1f5f9" },
    dataLabels: { enabled: false },
  };

  if (chartLocalShap) {
    chartLocalShap.destroy();
  }
  chartLocalShap = new ApexCharts(chartEl, options);
  chartLocalShap.render();
}

// ==============================================================================
// VIEW 3: INTERVENTIONS
// ==============================================================================

async function loadInterventions() {
  const tbody = document.getElementById("interventions-table-body");
  tbody.innerHTML = "";

  const highCases = allStudents.filter((s) => s.predicted_risk_tier === "High Risk" || s.predicted_risk_tier === "Medium Risk").slice(0, 15);

  highCases.forEach((s) => {
    let protocol = s.absences >= 10 ? "Attendance Recovery Contract" : s.failures > 0 ? "Prerequisite Remedial Foundation" : "Study Habits & Peer Circle";
    let priority = s.predicted_risk_tier === "High Risk" ? "Urgent" : "Moderate";
    let pBadge = priority === "Urgent" ? "bg-red-100 text-red-800" : "bg-amber-100 text-amber-800";

    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-50/80 transition";
    tr.innerHTML = `
      <td class="px-4 py-2.5 font-bold text-slate-800">${s.student_id}</td>
      <td class="px-4 py-2.5"><span class="${s.predicted_risk_tier === 'High Risk' ? 'badge-high' : 'badge-med'}">${s.predicted_risk_tier}</span></td>
      <td class="px-4 py-2.5"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${pBadge}">${priority}</span></td>
      <td class="px-4 py-2.5 text-slate-700 font-medium">${protocol}</td>
      <td class="px-4 py-2.5"><span class="text-xs text-blue-600 font-semibold cursor-pointer" onclick="openStudentModal('${s.student_id}')">Open Case ➔</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ==============================================================================
// VIEW 4: AI INSIGHTS (GLOBAL SHAP & ELEMENT DECOMPOSITION)
// ==============================================================================

async function loadInsights() {
  try {
    const res = await fetch(`${API_BASE}/api/ai-insights?mode=${currentMode}&dataset=${currentDataset}`);
    const data = await res.json();

    renderGlobalShapChart(data.feature_importance || []);
    renderElementsChart(data.sal_vs_ec || { sal_percentage: 67, ec_percentage: 33 });
  } catch (err) {
    console.error("Failed to load AI insights:", err);
  }
}

function renderGlobalShapChart(features) {
  const chartEl = document.querySelector("#chart-global-shap");
  if (!chartEl) return;

  const topFeats = features.slice(0, 10).reverse();
  const categories = topFeats.map((f) => f.feature);
  const seriesData = topFeats.map((f) => parseFloat((f.importance || 0).toFixed(3)));

  const options = {
    chart: {
      type: "bar",
      height: 300,
      toolbar: { show: false },
      fontFamily: "Plus Jakarta Sans, sans-serif",
    },
    plotOptions: {
      bar: {
        horizontal: true,
        borderRadius: 4,
        barHeight: "60%",
      },
    },
    colors: ["#2563eb"],
    series: [{ name: "Sensitivity", data: seriesData }],
    xaxis: {
      categories: categories,
      labels: { style: { colors: "#64748b", fontSize: "10px" } },
    },
    yaxis: {
      labels: { style: { colors: "#334155", fontSize: "11px", fontWeight: 500 } },
    },
    grid: { borderColor: "#f1f5f9" },
    dataLabels: { enabled: false },
  };

  if (chartGlobalShap) {
    chartGlobalShap.destroy();
  }
  chartGlobalShap = new ApexCharts(chartEl, options);
  chartGlobalShap.render();
}

function renderElementsChart(salVsEc) {
  const chartEl = document.querySelector("#chart-pie-elements");
  if (!chartEl) return;

  const sal = salVsEc.sal_percentage || 65;
  const ec = salVsEc.ec_percentage || 35;

  const options = {
    chart: {
      type: "donut",
      height: 240,
      fontFamily: "Plus Jakarta Sans, sans-serif",
    },
    labels: ["Subjective Behaviors", "Objective Context"],
    series: [sal, ec],
    colors: ["#2563eb", "#8b5cf6"],
    legend: { position: "bottom", labels: { colors: "#64748b" } },
    dataLabels: { enabled: false },
    plotOptions: {
      pie: {
        donut: { size: "65%" },
      },
    },
  };

  if (chartElements) {
    chartElements.destroy();
  }
  chartElements = new ApexCharts(chartEl, options);
  chartElements.render();
}

// ==============================================================================
// VIEW 5: BENCHMARKS
// ==============================================================================

let currentBmTab = "reg";
let benchmarkData = {};

async function loadBenchmarks() {
  try {
    const res = await fetch(`${API_BASE}/api/benchmarks?mode=${currentMode}&dataset=${currentDataset}`);
    benchmarkData = await res.json();
    renderBenchmarkContent();
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

function switchBenchmarkTab(tab) {
  currentBmTab = tab;
  ["reg", "bin", "risk"].forEach((t) => {
    const btn = document.getElementById(`bm-tab-${t}`);
    if (btn) {
      if (t === tab) {
        btn.className = "text-xs font-bold text-blue-600 border-b-2 border-blue-600 pb-2";
      } else {
        btn.className = "text-xs font-semibold text-slate-500 hover:text-slate-800 pb-2";
      }
    }
  });
  renderBenchmarkContent();
}

function renderBenchmarkContent() {
  const container = document.getElementById("benchmark-content");
  if (!container) return;

  if (currentBmTab === "reg") {
    const list = benchmarkData.regression || [];
    container.innerHTML = `
      <table class="w-full text-left text-xs">
        <thead class="bg-slate-50 text-slate-500 font-semibold uppercase">
          <tr>
            <th class="px-4 py-2.5">Model Family</th>
            <th class="px-4 py-2.5">RMSE (Lower is Better)</th>
            <th class="px-4 py-2.5">MAE</th>
            <th class="px-4 py-2.5">R² (Variance Explained)</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-100 font-medium">
          ${list.map((m) => `
            <tr class="${m.model_name.includes('Stacking') ? 'bg-blue-50/50 font-bold' : ''}">
              <td class="px-4 py-2 text-slate-900">${m.model_name}</td>
              <td class="px-4 py-2">${(m.RMSE || 0).toFixed(3)}</td>
              <td class="px-4 py-2">${(m.MAE || 0).toFixed(3)}</td>
              <td class="px-4 py-2 text-emerald-600">${(m.R2 || 0).toFixed(3)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  } else if (currentBmTab === "bin") {
    const list = benchmarkData.binary || [];
    container.innerHTML = `
      <table class="w-full text-left text-xs">
        <thead class="bg-slate-50 text-slate-500 font-semibold uppercase">
          <tr>
            <th class="px-4 py-2.5">Model Family</th>
            <th class="px-4 py-2.5">Accuracy</th>
            <th class="px-4 py-2.5">Precision</th>
            <th class="px-4 py-2.5">Recall</th>
            <th class="px-4 py-2.5">F1-Score</th>
            <th class="px-4 py-2.5">ROC-AUC</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-100 font-medium">
          ${list.map((m) => `
            <tr class="${m.model_name.includes('Stacking') ? 'bg-blue-50/50 font-bold' : ''}">
              <td class="px-4 py-2 text-slate-900">${m.model_name}</td>
              <td class="px-4 py-2">${(m.Accuracy || 0).toFixed(3)}</td>
              <td class="px-4 py-2">${(m.Precision || 0).toFixed(3)}</td>
              <td class="px-4 py-2">${(m.Recall || 0).toFixed(3)}</td>
              <td class="px-4 py-2 font-bold text-blue-600">${(m.F1 || 0).toFixed(3)}</td>
              <td class="px-4 py-2 text-emerald-600">${(m.ROC_AUC || 0).toFixed(3)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  } else {
    const list = benchmarkData.risk || [];
    container.innerHTML = `
      <table class="w-full text-left text-xs">
        <thead class="bg-slate-50 text-slate-500 font-semibold uppercase">
          <tr>
            <th class="px-4 py-2.5">Model Family</th>
            <th class="px-4 py-2.5">Accuracy</th>
            <th class="px-4 py-2.5">Macro-F1 (Imbalance Sensitive)</th>
            <th class="px-4 py-2.5">Weighted-F1</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-100 font-medium">
          ${list.map((m) => `
            <tr class="${m.model_name.includes('Stacking') ? 'bg-blue-50/50 font-bold' : ''}">
              <td class="px-4 py-2 text-slate-900">${m.model_name}</td>
              <td class="px-4 py-2">${(m.Accuracy || 0).toFixed(3)}</td>
              <td class="px-4 py-2 font-bold text-blue-600">${(m.Macro_F1 || 0).toFixed(3)}</td>
              <td class="px-4 py-2">${(m.Weighted_F1 || 0).toFixed(3)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
  }
}

// ==============================================================================
// VIEW 6: WHAT-IF SIMULATOR
// ==============================================================================

// ==============================================================================
// VIEW 6: WHAT-IF SIMULATOR (SELECT & MANUAL ENTRY MODES)
// ==============================================================================

let whatIfInputMode = "select"; // 'select' or 'manual'

function setWhatIfMode(mode) {
  whatIfInputMode = mode;
  const btnSelect = document.getElementById("btn-whatif-mode-select");
  const btnManual = document.getElementById("btn-whatif-mode-manual");
  const panelSelect = document.getElementById("whatif-select-panel");
  const panelManual = document.getElementById("whatif-manual-panel");

  if (mode === "select") {
    btnSelect.className = "px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white transition flex items-center gap-1.5";
    btnManual.className = "px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 transition flex items-center gap-1.5";
    panelSelect.classList.remove("hidden");
    panelManual.classList.add("hidden");

    const sel = document.getElementById("whatif-student-select");
    if (sel && sel.value) loadStudentIntoSimulator(sel.value);
  } else {
    btnManual.className = "px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white transition flex items-center gap-1.5";
    btnSelect.className = "px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 text-slate-600 hover:bg-slate-200 transition flex items-center gap-1.5";
    panelManual.classList.remove("hidden");
    panelSelect.classList.add("hidden");

    syncManualStudentBaseline();
  }
  lucide.createIcons();
}

function syncManualStudentBaseline() {
  const study = parseInt(document.getElementById("manual-study").value) || 2;
  const abs = parseInt(document.getElementById("manual-abs").value) || 12;
  const failures = parseInt(document.getElementById("manual-failures").value) || 1;
  const support = document.getElementById("manual-support").value || "no";
  const alcohol = parseInt(document.getElementById("manual-alcohol").value) || 1;

  // Preset proposed intervention sliders to smart initial targets
  document.getElementById("sim-study").value = Math.min(4, study + 1);
  updateSimValue("val-study", Math.min(4, study + 1));

  document.getElementById("sim-abs").value = Math.max(0, abs - 4);
  updateSimValue("val-abs", Math.max(0, abs - 4));

  document.getElementById("sim-failures").value = Math.max(0, failures - 1);
  updateSimValue("val-failures", Math.max(0, failures - 1));

  const supRadio = document.querySelector(`input[name="sim-support"][value="${support === 'yes' ? 'yes' : 'yes'}"]`);
  if (supRadio) supRadio.checked = true;

  document.getElementById("sim-dalc").value = Math.min(2, alcohol);
  updateSimValue("val-dalc", Math.min(2, alcohol));

  document.getElementById("sim-walc").value = Math.min(2, alcohol);
  updateSimValue("val-walc", Math.min(2, alcohol));
}

function populateSimulatorDropdown() {
  const select = document.getElementById("whatif-student-select");
  if (!select || allStudents.length === 0) return;

  select.innerHTML = allStudents.map((s) => `<option value="${s.student_id}">${s.student_id} (${s.predicted_risk_tier} • G1: ${s.G1 || '--'}/20)</option>`).join("");
  loadStudentIntoSimulator(allStudents[0].student_id);
}

function loadStudentIntoSimulator(studentId) {
  const s = allStudents.find((item) => item.student_id === studentId);
  if (!s) return;

  // Update student quick info banner
  const infoEl = document.getElementById("whatif-student-info");
  if (infoEl) {
    let bClass = s.predicted_risk_tier === "High Risk" ? "badge-high" : s.predicted_risk_tier === "Medium Risk" ? "badge-med" : "badge-low";
    infoEl.innerHTML = `
      <div class="flex items-center gap-1.5"><span class="font-semibold text-slate-700">ID:</span> <span class="font-bold text-slate-900">${s.student_id}</span></div>
      <div class="flex items-center gap-1.5"><span class="font-semibold text-slate-700">Risk Tier:</span> <span class="${bClass}">${s.predicted_risk_tier}</span></div>
      <div class="flex items-center gap-1.5"><span class="font-semibold text-slate-700">Early Mark (G1):</span> <span class="font-bold text-slate-800">${s.G1 || '--'}/20</span></div>
      <div class="flex items-center gap-1.5"><span class="font-semibold text-slate-700">Current Absences:</span> <span class="font-bold text-slate-800">${s.absences || 0} sessions</span></div>
      <div class="flex items-center gap-1.5"><span class="font-semibold text-slate-700">Study Time:</span> <span class="font-bold text-slate-800">Level ${s.studytime || 2}</span></div>
    `;
  }

  document.getElementById("sim-study").value = Math.min(4, (s.studytime || 2) + 1);
  updateSimValue("val-study", Math.min(4, (s.studytime || 2) + 1));

  document.getElementById("sim-abs").value = Math.max(0, (s.absences || 4) - 2);
  updateSimValue("val-abs", Math.max(0, (s.absences || 4) - 2));

  document.getElementById("sim-failures").value = Math.max(0, (s.failures || 0) - 1);
  updateSimValue("val-failures", Math.max(0, (s.failures || 0) - 1));

  document.getElementById("sim-dalc").value = Math.min(2, s.Dalc || 1);
  updateSimValue("val-dalc", Math.min(2, s.Dalc || 1));

  document.getElementById("sim-walc").value = Math.min(3, s.Walc || 1);
  updateSimValue("val-walc", Math.min(3, s.Walc || 1));
}

function updateSimValue(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  if (id === "val-study") el.innerText = `${val} (${val === "1" ? "< 2 hrs" : val === "2" ? "2–5 hrs" : val === "3" ? "5–10 hrs" : "> 10 hrs"})`;
  if (id === "val-abs") el.innerText = `${val} sessions`;
  if (id === "val-failures") el.innerText = `${val} cleared`;
  if (id === "val-dalc") el.innerText = `${val} (${val === "1" ? "Very Low" : val === "5" ? "High" : "Moderate"})`;
  if (id === "val-walc") el.innerText = `${val} (${val === "1" ? "Very Low" : val === "5" ? "High" : "Moderate"})`;
}

async function runSimulation() {
  const study = parseInt(document.getElementById("sim-study").value);
  const abs = parseInt(document.getElementById("sim-abs").value);
  const failures = parseInt(document.getElementById("sim-failures").value);
  const dalc = parseInt(document.getElementById("sim-dalc").value);
  const walc = parseInt(document.getElementById("sim-walc").value);
  const supportRadio = document.querySelector('input[name="sim-support"]:checked');
  const support = supportRadio ? supportRadio.value : "no";

  const loadingText = document.getElementById("sim-loading-text");
  const btn = document.getElementById("btn-run-sim");
  if (loadingText) loadingText.innerText = "Computing projected outcomes via ML Ensemble...";
  if (btn) btn.disabled = true;

  let payload = {
    mode: currentMode,
    dataset: currentDataset,
    modifications: {
      studytime: study,
      absences: abs,
      failures: failures,
      schoolsup: support,
      Dalc: dalc,
      Walc: walc,
    },
  };

  if (whatIfInputMode === "manual") {
    const g1 = parseFloat(document.getElementById("manual-g1").value) || 8.0;
    const baseAbs = parseInt(document.getElementById("manual-abs").value) || 12;
    const baseStudy = parseInt(document.getElementById("manual-study").value) || 2;
    const baseFail = parseInt(document.getElementById("manual-failures").value) || 1;
    const baseSup = document.getElementById("manual-support").value || "no";
    const baseAlc = parseInt(document.getElementById("manual-alcohol").value) || 1;

    payload.custom_profile = {
      student_id: "CUSTOM-STUDENT",
      G1: g1,
      absences: baseAbs,
      studytime: baseStudy,
      failures: baseFail,
      schoolsup: baseSup,
      Dalc: baseAlc,
      Walc: baseAlc,
    };
  } else {
    payload.student_id = document.getElementById("whatif-student-select").value;
  }

  try {
    const res = await fetch(`${API_BASE}/api/what-if`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Simulation Error: ${err.error || "Failed to run simulation."}`);
      return;
    }

    const result = await res.json();

    const box = document.getElementById("sim-results");
    box.classList.remove("hidden");

    // Student badge indicator
    const stuBadge = document.getElementById("sim-student-badge");
    if (stuBadge) {
      stuBadge.innerText = whatIfInputMode === "manual" ? "Profile: Custom Manual Student" : `Cohort Student: ${payload.student_id}`;
    }

    document.getElementById("sim-score").innerText = `${result.simulated_score.toFixed(1)} / 20`;
    const deltaSign = result.score_delta >= 0 ? "+" : "";
    document.getElementById("sim-delta").innerText = `${deltaSign}${result.score_delta.toFixed(2)} pts compared to baseline (${result.baseline_score.toFixed(1)})`;

    document.getElementById("sim-tier-shift").innerHTML = `
      <span class="text-slate-600">${result.baseline_risk}</span> ➔ <span class="text-emerald-600 font-extrabold">${result.simulated_risk}</span>
    `;

    const statusBox = document.getElementById("sim-status-box");
    if (result.risk_reduced) {
      statusBox.innerHTML = `
        <div class="text-xs text-emerald-800 bg-emerald-50 border border-emerald-200 rounded p-3 w-full">
          <b>✅ Effective Intervention:</b> Proposed plan successfully demotes academic risk tier!
        </div>
      `;
    } else {
      statusBox.innerHTML = `
        <div class="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded p-3 w-full">
          <b>⚠️ Marginal Improvement:</b> More intensive foundational tutoring is suggested.
        </div>
      `;
    }

    // Render Probability Shift Chart
    renderSimProbChart(result.baseline_probabilities, result.simulated_probabilities);

    // Render Actionable Suggestions List
    const suggList = document.getElementById("whatif-suggestions-list");
    if (suggList) {
      suggList.innerHTML = "";
      (result.suggestions || []).forEach((sug) => {
        let bColor = sug.type === "positive" || sug.type === "success" 
          ? "bg-emerald-50 border-emerald-200 text-emerald-900" 
          : sug.type === "warning" 
          ? "bg-amber-50 border-amber-200 text-amber-900" 
          : sug.type === "critical"
          ? "bg-red-50 border-red-200 text-red-900"
          : "bg-blue-50 border-blue-200 text-blue-900";
        
        let tagColor = sug.type === "positive" || sug.type === "success"
          ? "bg-emerald-100 text-emerald-800"
          : sug.type === "warning"
          ? "bg-amber-100 text-amber-800"
          : sug.type === "critical"
          ? "bg-red-100 text-red-800"
          : "bg-blue-100 text-blue-800";

        const div = document.createElement("div");
        div.className = `border rounded-lg p-3 text-xs space-y-1 ${bColor}`;
        div.innerHTML = `
          <div class="flex items-center justify-between">
            <span class="font-bold flex items-center gap-1.5">
              <span>${sug.title}</span>
            </span>
            <span class="text-[10px] uppercase font-bold px-2 py-0.5 rounded ${tagColor}">${sug.category}</span>
          </div>
          <p class="leading-relaxed opacity-95">${sug.text}</p>
        `;
        suggList.appendChild(div);
      });
    }

    box.scrollIntoView({ behavior: "smooth" });
    lucide.createIcons();
  } catch (err) {
    console.error("Simulation error:", err);
    alert(`Simulation Error: ${err.message}`);
  } finally {
    if (loadingText) loadingText.innerText = "";
    if (btn) btn.disabled = false;
  }
}

function renderSimProbChart(baseProbs, simProbs) {
  const chartEl = document.querySelector("#chart-sim-prob");
  if (!chartEl) return;

  const options = {
    chart: {
      type: "bar",
      height: 220,
      toolbar: { show: false },
      fontFamily: "Plus Jakarta Sans, sans-serif",
    },
    plotOptions: {
      bar: { horizontal: false, columnWidth: "45%", borderRadius: 4 },
    },
    series: [
      { name: "Pre-Intervention", data: baseProbs },
      { name: "Post-Intervention", data: simProbs },
    ],
    colors: ["#94a3b8", "#2563eb"],
    xaxis: {
      categories: ["Low Risk", "Medium Risk", "High Risk"],
      labels: { style: { colors: "#64748b", fontSize: "11px" } },
    },
    yaxis: {
      max: 1.0,
      labels: { style: { colors: "#64748b", fontSize: "10px" } },
    },
    grid: { borderColor: "#f1f5f9" },
    dataLabels: { enabled: false },
  };

  if (chartSimProb) {
    chartSimProb.destroy();
  }
  chartSimProb = new ApexCharts(chartEl, options);
  chartSimProb.render();
}

// ==============================================================================
// VIEW 7: UPLOAD DATA
// ==============================================================================

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const btn = document.getElementById("btn-browse-file");
  const statusEl = document.getElementById("upload-status");
  const resultBox = document.getElementById("upload-results");

  if (statusEl) {
    statusEl.className = "text-xs font-semibold text-blue-600 block mt-2";
    statusEl.innerText = `⏳ Uploading and scoring '${file.filename || file.name}'...`;
  }
  if (btn) btn.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/api/upload?mode=${currentMode}&dataset=${currentDataset}`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      let errMsg = "Upload failed";
      try {
        const errJson = await res.json();
        errMsg = errJson.error || errJson.detail || errMsg;
      } catch (e) {
        errMsg = await res.text();
      }
      if (statusEl) {
        statusEl.className = "text-xs font-semibold text-red-600 block mt-2";
        statusEl.innerText = `❌ Error: ${errMsg}`;
      }
      if (btn) btn.disabled = false;
      return;
    }

    const result = await res.json();

    if (statusEl) {
      statusEl.className = "text-xs font-semibold text-emerald-600 block mt-2";
      statusEl.innerText = `✅ Successfully scored ${result.total_records} student records!`;
    }

    if (resultBox) resultBox.classList.remove("hidden");

    document.getElementById("upload-summary").innerText = `Scored Cohort Records (${result.total_records} students from '${result.filename}')`;

    const tbody = document.getElementById("upload-table-body");
    tbody.innerHTML = "";

    (result.scored_records || []).forEach((row) => {
      let bClass = row.predicted_risk_tier === "High Risk" ? "badge-high" : row.predicted_risk_tier === "Medium Risk" ? "badge-med" : "badge-low";
      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-50 transition";
      tr.innerHTML = `
        <td class="px-4 py-2.5 font-bold text-slate-800">${row.student_id}</td>
        <td class="px-4 py-2.5 font-bold text-slate-900">${(row.predicted_final_score || 0).toFixed(1)} / 20</td>
        <td class="px-4 py-2.5"><span class="${bClass}">${row.predicted_risk_tier}</span></td>
        <td class="px-4 py-2.5 text-slate-600">${row.studytime || "--"}</td>
        <td class="px-4 py-2.5 text-slate-600">${row.absences !== undefined ? row.absences : "--"}</td>
        <td class="px-4 py-2.5 text-slate-600">${row.failures !== undefined ? row.failures : 0}</td>
      `;
      tbody.appendChild(tr);
    });

    if (resultBox) resultBox.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    console.error("Upload scoring failed:", err);
    if (statusEl) {
      statusEl.className = "text-xs font-semibold text-red-600 block mt-2";
      statusEl.innerText = `❌ Upload failed: ${err.message}`;
    }
  } finally {
    if (btn) btn.disabled = false;
    event.target.value = "";
  }
}
