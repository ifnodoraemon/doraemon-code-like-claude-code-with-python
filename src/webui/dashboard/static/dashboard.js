/**
 * Doraemon Code Evaluation Dashboard JavaScript
 *
 * Handles chart rendering, data fetching, and UI interactions.
 */

// =============================================================================
// Chart Initialization
// =============================================================================

let successRateChart = null;
let latencyChart = null;
let difficultyChart = null;

const API_KEY_STORAGE_KEY = 'doraemon.webui.apiKey';
let apiKeyPromptPromise = null;

function getStoredApiKey() {
    try {
        return window.localStorage.getItem(API_KEY_STORAGE_KEY) || '';
    } catch (error) {
        return '';
    }
}

function setStoredApiKey(apiKey) {
    try {
        const normalizedKey = apiKey.trim();
        if (normalizedKey) {
            window.localStorage.setItem(API_KEY_STORAGE_KEY, normalizedKey);
        } else {
            window.localStorage.removeItem(API_KEY_STORAGE_KEY);
        }
    } catch (error) {
        // Storage can be disabled; auth failures will still be surfaced by the API.
    }
}

function isProtectedApiRequest(input) {
    try {
        const url = new URL(typeof input === 'string' ? input : input.url, window.location.origin);
        return url.origin === window.location.origin
            && (url.pathname === '/dashboard/api'
                || url.pathname.startsWith('/dashboard/api/')
                || url.pathname === '/api'
                || url.pathname.startsWith('/api/'));
    } catch (error) {
        return false;
    }
}

function withDashboardAuth(input, init = {}) {
    if (!isProtectedApiRequest(input)) {
        return init;
    }

    const apiKey = getStoredApiKey();
    if (!apiKey) {
        return init;
    }

    const headers = new Headers(init.headers || {});
    if (!headers.has('Authorization')) {
        headers.set('Authorization', `Bearer ${apiKey}`);
    }
    return { ...init, headers };
}

async function dashboardFetch(input, init) {
    let response = await fetch(input, withDashboardAuth(input, init));
    if (!isProtectedApiRequest(input) || ![401, 503].includes(response.status)) {
        return response;
    }

    const apiKey = await promptForApiKey();
    if (apiKey === null) {
        return response;
    }

    setStoredApiKey(apiKey);
    response = await fetch(input, withDashboardAuth(input, init));
    return response;
}

async function promptForApiKey() {
    if (!apiKeyPromptPromise) {
        apiKeyPromptPromise = Promise.resolve()
            .then(() => window.prompt('API key required'))
            .finally(() => {
                apiKeyPromptPromise = null;
            });
    }
    return apiKeyPromptPromise;
}

/**
 * Initialize all charts on page load.
 */
function initializeCharts() {
    [successRateChart, latencyChart, difficultyChart].forEach(chart => {
        if (chart) {
            chart.destroy();
        }
    });

    initSuccessRateChart();
    initLatencyChart();
    initDifficultyChart();
}

/**
 * Initialize the success rate trend chart.
 */
function initSuccessRateChart() {
    const ctx = document.getElementById('successRateChart');
    if (!ctx) return;

    const data = trendsData.success_rate_trend || [];
    const labels = data.map(d => d.label);
    const values = data.map(d => d.value);

    successRateChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Success Rate (%)',
                data: values,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Success Rate: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Initialize the latency distribution chart.
 */
function initLatencyChart() {
    const ctx = document.getElementById('latencyChart');
    if (!ctx) return;

    const data = trendsData.latency_trend || [];
    const labels = data.map(d => d.label);
    const values = data.map(d => d.value);

    latencyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Avg Latency (s)',
                data: values,
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: '#3b82f6',
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Latency: ${context.parsed.y.toFixed(2)}s`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value + 's';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Initialize the difficulty analysis chart.
 */
function initDifficultyChart() {
    const ctx = document.getElementById('difficultyChart');
    if (!ctx) return;

    const byDifficulty = taskStats.by_difficulty || {};
    const labels = Object.keys(byDifficulty);
    const successRates = labels.map(d => (byDifficulty[d].success_rate || 0) * 100);
    const totals = labels.map(d => byDifficulty[d].total || 0);

    // Color mapping for difficulty levels
    const colorMap = {
        'easy': '#10b981',
        'medium': '#f59e0b',
        'hard': '#ef4444',
        'expert': '#7c3aed',
    };

    const colors = labels.map(d => colorMap[d.toLowerCase()] || '#64748b');

    difficultyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
            datasets: [{
                label: 'Success Rate (%)',
                data: successRates,
                backgroundColor: colors.map(c => c + 'cc'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const idx = context.dataIndex;
                            const total = totals[idx];
                            return [
                                `Success Rate: ${context.parsed.y.toFixed(1)}%`,
                                `Total Tasks: ${total}`
                            ];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}

// =============================================================================
// Data Loading
// =============================================================================

/**
 * Load model comparison data and populate the table.
 */
async function loadModelComparison() {
    try {
        const response = await dashboardFetch('/dashboard/api/models/compare');
        const data = await response.json();

        const tbody = document.querySelector('#modelComparisonTable tbody');
        if (!tbody) return;

        tbody.innerHTML = '';

        const models = data.models || {};
        for (const [model, stats] of Object.entries(models)) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${escapeHtml(model)}</strong></td>
                <td>${stats.evaluations}</td>
                <td>${stats.total_tasks}</td>
                <td>
                    <div class="success-rate-bar">
                        <div class="bar-fill" style="width: ${stats.success_rate * 100}%"></div>
                        <span class="bar-label">${(stats.success_rate * 100).toFixed(1)}%</span>
                    </div>
                </td>
                <td>${stats.avg_time_per_task.toFixed(2)}s</td>
            `;
            tbody.appendChild(row);
        }

        if (Object.keys(models).length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">No data available</td></tr>';
        }
    } catch (error) {
        console.error('Failed to load model comparison:', error);
    }
}

/**
 * Load dashboard data through authenticated API requests and update the page.
 */
async function loadDashboardData() {
    try {
        const [tasksResponse, trendsResponse, evaluationsResponse] = await Promise.all([
            dashboardFetch('/dashboard/api/tasks'),
            dashboardFetch('/dashboard/api/trends'),
            dashboardFetch('/dashboard/api/evaluations?limit=10'),
        ]);

        if (!tasksResponse.ok || !trendsResponse.ok || !evaluationsResponse.ok) {
            return;
        }

        taskStats = await tasksResponse.json();
        trendsData = await trendsResponse.json();
        const evaluationsData = await evaluationsResponse.json();

        renderSummaryCards(taskStats);
        renderCategoryHeatmap(taskStats.by_category || {});
        renderCategoryFilter(taskStats.by_category || {});
        renderEvaluations(evaluationsData.evaluations || []);
        initializeCharts();
        await loadModelComparison();
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
    }
}

function renderSummaryCards(stats) {
    const successRate = document.getElementById('overallSuccessRate');
    const totalTasks = document.getElementById('totalTasks');
    const totalEvaluations = document.getElementById('totalEvaluations');
    const totalCategories = document.getElementById('totalCategories');

    if (successRate) {
        successRate.textContent = `${((stats.overall_success_rate || 0) * 100).toFixed(1)}%`;
    }
    if (totalTasks) {
        totalTasks.textContent = String(stats.total_tasks || 0);
    }
    if (totalEvaluations) {
        totalEvaluations.textContent = String(stats.total_evaluations || 0);
    }
    if (totalCategories) {
        totalCategories.textContent = String(Object.keys(stats.by_category || {}).length);
    }
}

function renderCategoryHeatmap(byCategory) {
    const container = document.getElementById('categoryHeatmap');
    if (!container) return;

    container.innerHTML = '';
    for (const [category, stats] of Object.entries(byCategory)) {
        const cell = document.createElement('div');
        const successRate = stats.success_rate || 0;
        cell.className = 'heatmap-cell';
        cell.style.setProperty('--success-rate', String(successRate));
        cell.title = `${category}: ${(successRate * 100).toFixed(1)}% (${stats.success || 0}/${stats.total || 0})`;

        const label = document.createElement('span');
        label.className = 'cell-label';
        label.textContent = category;
        const value = document.createElement('span');
        value.className = 'cell-value';
        value.textContent = `${(successRate * 100).toFixed(0)}%`;

        cell.append(label, value);
        container.appendChild(cell);
    }
}

function renderCategoryFilter(byCategory) {
    const select = document.getElementById('categoryFilter');
    if (!select) return;

    select.innerHTML = '';
    const allOption = document.createElement('option');
    allOption.value = '';
    allOption.textContent = 'All Categories';
    select.appendChild(allOption);

    for (const category of Object.keys(byCategory)) {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        select.appendChild(option);
    }
}

function renderEvaluations(evaluations) {
    const tbody = document.querySelector('#evaluationsTable tbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    for (const evaluation of evaluations) {
        const row = document.createElement('tr');
        row.dataset.evalId = evaluation.id;

        const timestamp = document.createElement('td');
        timestamp.textContent = String(evaluation.timestamp || '').slice(0, 19);

        const categoryCell = document.createElement('td');
        const categoryBadge = document.createElement('span');
        categoryBadge.className = 'category-badge';
        categoryBadge.textContent = evaluation.category || 'unknown';
        categoryCell.appendChild(categoryBadge);

        const tasks = document.createElement('td');
        tasks.textContent = String(evaluation.total_tasks || 0);

        const success = document.createElement('td');
        const successRate = (evaluation.success_rate || 0) * 100;
        success.innerHTML = `
            <div class="success-rate-bar">
                <div class="bar-fill" style="width: ${successRate}%"></div>
                <span class="bar-label">${successRate.toFixed(1)}%</span>
            </div>
        `;

        const duration = document.createElement('td');
        duration.textContent = `${(evaluation.total_time || 0).toFixed(2)}s`;

        const actions = document.createElement('td');
        const button = document.createElement('button');
        button.className = 'btn btn-small btn-view';
        button.dataset.evalId = evaluation.id;
        button.textContent = 'View';
        button.addEventListener('click', () => viewDetails(button.dataset.evalId));
        actions.appendChild(button);

        row.append(timestamp, categoryCell, tasks, success, duration, actions);
        tbody.appendChild(row);
    }
}

/**
 * Refresh all dashboard data.
 */
async function refreshDashboard() {
    try {
        await loadDashboardData();
    } catch (error) {
        console.error('Failed to refresh dashboard:', error);
        showNotification('Failed to refresh dashboard', 'error');
    }
}

// =============================================================================
// Evaluation Details
// =============================================================================

/**
 * View detailed evaluation results.
 * @param {string} evalId - Evaluation ID
 */
async function viewDetails(evalId) {
    try {
        const response = await dashboardFetch(`/dashboard/api/evaluations/${evalId}`);
        if (!response.ok) {
            throw new Error('Failed to load evaluation details');
        }

        const data = await response.json();
        renderDetailModal(data);
        openModal('detailModal');
    } catch (error) {
        console.error('Failed to load evaluation details:', error);
        showNotification('Failed to load evaluation details', 'error');
    }
}

/**
 * Render the detail modal content.
 * @param {Object} data - Evaluation data
 */
function renderDetailModal(data) {
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');

    if (!modalTitle || !modalBody) return;

    const summary = data.summary || {};
    const results = data.results || [];

    modalTitle.textContent = `Evaluation: ${data.id}`;

    // Build summary section
    let html = `
        <div class="detail-summary">
            <div class="detail-stat">
                <span class="stat-value">${summary.total_tasks || 0}</span>
                <span class="stat-label">Total Tasks</span>
            </div>
            <div class="detail-stat">
                <span class="stat-value">${((summary.success_rate || 0) * 100).toFixed(1)}%</span>
                <span class="stat-label">Success Rate</span>
            </div>
            <div class="detail-stat">
                <span class="stat-value">${(summary.total_time || 0).toFixed(2)}s</span>
                <span class="stat-label">Total Time</span>
            </div>
            <div class="detail-stat">
                <span class="stat-value">${(summary.avg_time_per_task || 0).toFixed(2)}s</span>
                <span class="stat-label">Avg Time/Task</span>
            </div>
        </div>
    `;

    // Build category breakdown
    const byCategory = summary.by_category || {};
    if (Object.keys(byCategory).length > 0) {
        html += `
            <div class="detail-section">
                <h4>By Category</h4>
                <div class="task-list">
        `;
        for (const [category, stats] of Object.entries(byCategory)) {
            const successRate = stats.total > 0 ? (stats.success / stats.total * 100) : 0;
            html += `
                <div class="task-item">
                    <div class="task-info">
                        <span class="task-status ${successRate >= 50 ? 'success' : 'failed'}"></span>
                        <span class="task-name">${escapeHtml(category)}</span>
                    </div>
                    <div class="task-meta">
                        <span>${stats.success}/${stats.total} passed</span>
                        <span>${successRate.toFixed(1)}%</span>
                    </div>
                </div>
            `;
        }
        html += '</div></div>';
    }

    // Build task results
    if (results.length > 0) {
        html += `
            <div class="detail-section">
                <h4>Task Results</h4>
                <div class="task-list">
        `;
        for (const result of results) {
            html += `
                <div class="task-item">
                    <div class="task-info">
                        <span class="task-status ${result.success ? 'success' : 'failed'}"></span>
                        <span class="task-name">${escapeHtml(result.task_id || result.task_name)}</span>
                    </div>
                    <div class="task-meta">
                        <span class="category-badge">${escapeHtml(result.category || 'unknown')}</span>
                        <span>${escapeHtml(result.difficulty || 'unknown')}</span>
                        <span>${(result.execution_time || 0).toFixed(2)}s</span>
                    </div>
                </div>
            `;
        }
        html += '</div></div>';
    }

    modalBody.innerHTML = html;
}

// =============================================================================
// New Evaluation
// =============================================================================

/**
 * Open the new evaluation modal.
 */
function openNewEvalModal() {
    openModal('newEvalModal');
}

/**
 * Close the new evaluation modal.
 */
function closeNewEvalModal() {
    closeModal('newEvalModal');
}

/**
 * Submit a new evaluation request.
 * @param {Event} event - Form submit event
 */
async function submitNewEvaluation(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const request = {
        task_set: formData.get('task_set'),
        n_trials: parseInt(formData.get('n_trials')) || 1,
        max_workers: parseInt(formData.get('max_workers')) || 2,
        model: formData.get('model') || null,
    };

    try {
        const response = await dashboardFetch('/dashboard/api/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            throw new Error('Failed to start evaluation');
        }

        const data = await response.json();
        showNotification(`Evaluation started: ${data.id}`, 'success');
        closeNewEvalModal();

        // Start polling for progress
        pollEvaluationProgress(data.id);
    } catch (error) {
        console.error('Failed to start evaluation:', error);
        showNotification('Failed to start evaluation', 'error');
    }
}

/**
 * Poll for evaluation progress.
 * @param {string} evalId - Evaluation ID
 */
async function pollEvaluationProgress(evalId) {
    const pollInterval = 2000; // 2 seconds

    const poll = async () => {
        try {
            const response = await dashboardFetch(`/dashboard/api/evaluate/${evalId}/progress`);
            if (!response.ok) {
                throw new Error('Failed to get progress');
            }

            const progress = await response.json();
            updateProgressUI(progress);

            if (progress.status === 'running' || progress.status === 'pending') {
                setTimeout(poll, pollInterval);
            } else if (progress.status === 'completed') {
                showNotification('Evaluation completed!', 'success');
                setTimeout(refreshDashboard, 1000);
            } else if (progress.status === 'failed') {
                showNotification(`Evaluation failed: ${progress.error}`, 'error');
            }
        } catch (error) {
            console.error('Failed to poll progress:', error);
        }
    };

    poll();
}

/**
 * Update the progress UI for a running evaluation.
 * @param {Object} progress - Progress data
 */
function updateProgressUI(progress) {
    const card = document.querySelector(`.progress-card[data-eval-id="${progress.id}"]`);
    if (!card) {
        // Create new progress card if it doesn't exist
        const container = document.querySelector('.running-evaluations');
        if (container) {
            const newCard = document.createElement('div');
            newCard.className = 'progress-card';
            newCard.dataset.evalId = progress.id;
            newCard.innerHTML = `
                <div class="progress-header">
                    <span class="eval-id">${progress.id}</span>
                    <span class="eval-status status-${progress.status}">${progress.status}</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: ${progress.progress * 100}%"></div>
                </div>
                <div class="progress-details">
                    <span>${progress.completed_tasks}/${progress.total_tasks} tasks</span>
                    ${progress.current_task ? `<span>Current: ${progress.current_task}</span>` : ''}
                </div>
            `;
            container.appendChild(newCard);
        }
        return;
    }

    // Update existing card
    const statusEl = card.querySelector('.eval-status');
    const progressBar = card.querySelector('.progress-bar');
    const detailsEl = card.querySelector('.progress-details');

    if (statusEl) {
        statusEl.textContent = progress.status;
        statusEl.className = `eval-status status-${progress.status}`;
    }

    if (progressBar) {
        progressBar.style.width = `${progress.progress * 100}%`;
    }

    if (detailsEl) {
        detailsEl.innerHTML = `
            <span>${progress.completed_tasks}/${progress.total_tasks} tasks</span>
            ${progress.current_task ? `<span>Current: ${progress.current_task}</span>` : ''}
        `;
    }
}

// =============================================================================
// Modal Utilities
// =============================================================================

/**
 * Open a modal by ID.
 * @param {string} modalId - Modal element ID
 */
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('active');
    }
}

/**
 * Close a modal by ID or the detail modal by default.
 * @param {string} [modalId='detailModal'] - Modal element ID
 */
function closeModal(modalId = 'detailModal') {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('active');
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Escape HTML to prevent XSS.
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show a notification message.
 * @param {string} message - Notification message
 * @param {string} type - Notification type ('success', 'error', 'info')
 */
function showNotification(message, type = 'info') {
    // Simple alert for now - could be replaced with a toast library
    if (type === 'error') {
        console.error(message);
    } else {
        console.log(message);
    }

    // Create toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 24px;
        background-color: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 9999;
        animation: slideIn 0.3s ease;
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Filter evaluations by category.
 * @param {string} category - Category to filter by
 */
function filterByCategory(category) {
    const rows = document.querySelectorAll('#evaluationsTable tbody tr');
    rows.forEach(row => {
        const rowCategory = row.querySelector('.category-badge')?.textContent;
        if (!category || rowCategory === category) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// =============================================================================
// Event Listeners
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts
    initializeCharts();
    void loadDashboardData();

    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshDashboard);
    }

    // New evaluation button
    const newEvalBtn = document.getElementById('newEvalBtn');
    if (newEvalBtn) {
        newEvalBtn.addEventListener('click', openNewEvalModal);
    }

    // New evaluation form
    const newEvalForm = document.getElementById('newEvalForm');
    if (newEvalForm) {
        newEvalForm.addEventListener('submit', submitNewEvaluation);
    }

    // Category filter
    const categoryFilter = document.getElementById('categoryFilter');
    if (categoryFilter) {
        categoryFilter.addEventListener('change', (e) => {
            filterByCategory(e.target.value);
        });
    }

    // Close modal on outside click
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    });

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal.active').forEach(modal => {
                modal.classList.remove('active');
            });
        }
    });
});

// Add CSS animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
