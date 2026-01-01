// App Logic

const API_BASE = 'http://localhost:8000/api';

// State
let currentRunData = null;
let currentChart = null;

// DOM Elements
const views = {
    dashboard: document.getElementById('view-dashboard'),
    analysis: document.getElementById('view-analysis'),
    losers: document.getElementById('view-losers')
};
const navBtns = document.querySelectorAll('.nav-btn');
const runBtn = document.getElementById('run-analysis-btn');
const refreshBtn = document.getElementById('refresh-btn');
const modal = document.getElementById('graph-modal');
const closeModal = document.querySelector('.close-modal');

// Init
async function init() {
    setupNavigation();
    setupEventListeners();
    await fetchLatestRun();
}

// Navigation
function setupNavigation() {
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active state
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Show view
            const viewName = btn.dataset.view;
            Object.values(views).forEach(v => v.classList.remove('active'));
            views[viewName].classList.add('active');
        });
    });
}

// Event Listeners
function setupEventListeners() {
    runBtn.addEventListener('click', runAnalysis);
    refreshBtn.addEventListener('click', fetchLatestRun);
    closeModal.addEventListener('click', () => modal.classList.add('hidden'));

    // Close modal on outside click
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.classList.add('hidden');
    });
}

// API Calls
async function fetchLatestRun() {
    showMessage('Refreshing data...', 'info');
    try {
        const response = await fetch(`${API_BASE}/results/latest`);
        if (!response.ok) throw new Error('No run data found');

        const data = await response.json();
        currentRunData = data;

        renderDashboard(data);
        renderAnalysis(data);
        renderLosers(data);

        document.getElementById('latest-run-display').textContent = `Run: ${data.run_id}`;
        showMessage('Data loaded successfully', 'success');

    } catch (err) {
        console.error(err);
        document.getElementById('latest-run-display').textContent = 'No Data Available';
        showMessage('Failed to load data', 'error');
    }
}

async function runAnalysis() {
    const statusDiv = document.getElementById('analysis-status');
    const statusText = document.getElementById('status-text');

    statusDiv.classList.remove('hidden');
    statusText.textContent = 'Starting Analysis Workflow...';
    runBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE}/analyze`, { method: 'POST' });
        const res = await response.json();

        statusText.textContent = 'Analysis Running in Background...';
        setTimeout(() => {
            statusDiv.classList.add('hidden');
            runBtn.disabled = false;
            fetchLatestRun(); // Try to refresh
        }, 5000); // Simulate wait, real workflow takes longer

    } catch (err) {
        statusText.textContent = 'Error starting analysis';
        runBtn.disabled = false;
    }
}

// Rendering
function renderDashboard(data) {
    const longs = data.analysis_report?.length || 0;
    const shorts = data.downside_report?.length || 0;
    const total = data.stock_losers?.length || 0;

    document.getElementById('stat-longs').textContent = longs;
    document.getElementById('stat-shorts').textContent = shorts;
    document.getElementById('stat-total').textContent = total;
}

function renderAnalysis(data) {
    const longsContainer = document.getElementById('longs-list');
    const shortsContainer = document.getElementById('shorts-list');

    longsContainer.innerHTML = '';
    shortsContainer.innerHTML = '';

    // Render Longs
    if (data.analysis_report && Array.isArray(data.analysis_report)) {
        data.analysis_report.forEach(item => {
            const card = createStockCard(item, 'long');
            longsContainer.appendChild(card);
        });
    }

    // Render Shorts
    if (data.downside_report && Array.isArray(data.downside_report)) {
        data.downside_report.forEach(item => {
            const card = createStockCard(item, 'short');
            shortsContainer.appendChild(card);
        });
    }
}

function createStockCard(item, type) {
    const div = document.createElement('div');
    div.className = `stock-card glass-panel ${type}`;

    // Check if we have timing data matching this ticker
    // timing_output / downside_timing_output
    let action = "Hold";
    let score = "N/A";

    if (type === 'long') {
        const timing = currentRunData.timing_output?.find(t => t.ticker === item.ticker);
        action = timing ? timing.action : "Rebound Watch";
        score = item.rebound_probability || "0.0";
    } else {
        const timing = currentRunData.downside_timing_output?.find(t => t.ticker === item.ticker);
        action = timing ? timing.action : "Downside Watch";
        score = item.downside_probability || "0.0";
    }

    div.innerHTML = `
        <div class="card-header">
            <span class="ticker">${item.ticker}</span>
            <span class="action-badge">${action}</span>
        </div>
        <div class="card-details">
            <div class="detail-item">Price: <span>$${item.price || 'N/A'}</span></div>
            <div class="detail-item">Sector: <span>${item.sector || 'N/A'}</span></div>
        </div>
        <div class="pushout-box">
            <span class="pushout-score">${score}</span>
            <span class="pushout-label">Prob</span>
        </div>
    `;

    div.addEventListener('click', () => openStockModal(item.ticker));

    return div;
}

function renderLosers(data) {
    const tbody = document.querySelector('#losers-table tbody');
    tbody.innerHTML = '';

    if (data.stock_losers && Array.isArray(data.stock_losers)) {
        data.stock_losers.forEach(loser => {
            const tr = document.createElement('tr');

            const changeClass = parseFloat(loser.changesPercentage) >= 0 ? 'change-pos' : 'change-neg';

            tr.innerHTML = `
                <td><strong>${loser.symbol}</strong></td>
                <td>${loser.name}</td>
                <td>$${loser.price}</td>
                <td class="${changeClass}">${loser.changesPercentage}%</td>
                <td>${loser.volume?.toLocaleString()}</td>
                <td>${loser.sector || '-'}</td>
                <td><button class="icon-btn small-btn" onclick="openStockModal('${loser.symbol}')"><i class="fa-solid fa-chart-area"></i></button></td>
            `;
            // Note: inline onclick doesn't work easily with modules without attaching to window, 
            // but for simplicity we'll add event listener via query selector match or data attribute if this fails.
            // Better approach:

            tr.addEventListener('click', () => openStockModal(loser.symbol));
            tr.style.cursor = 'pointer';

            tbody.appendChild(tr);
        });
    }
}

// Charting
window.openStockModal = openStockModal; // Expose for inline calls if needed

async function openStockModal(ticker) {
    document.getElementById('modal-ticker-title').textContent = `${ticker} Analysis`;
    modal.classList.remove('hidden');

    // Clear previous chart
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }

    // Show loader in canvas area? Or just wait
    // Fetch History
    try {
        const response = await fetch(`${API_BASE}/stock/${ticker}/history`);
        const history = await response.json();

        renderChart(history, ticker);
    } catch (err) {
        console.error("Failed to load history", err);
    }
}

function renderChart(history, ticker) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    const labels = history.map(h => h.date);
    const dataPoints = history.map(h => h.close);

    // Gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(0, 255, 157, 0.5)');
    gradient.addColorStop(1, 'rgba(0, 255, 157, 0)');

    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `${ticker} Price`,
                data: dataPoints,
                borderColor: '#00ff9d',
                backgroundColor: gradient,
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(20, 22, 28, 0.9)',
                    titleColor: '#00ff9d',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8b9bb4' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8b9bb4' }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
}

function showMessage(msg, type) {
    console.log(`[${type}] ${msg}`);
}

// Start
init();
