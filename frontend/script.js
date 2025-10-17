/**
 * Simple PyData Assistant Frontend JavaScript
 * Basic functionality to test the application
 */

class PyDataAssistant {
    constructor() {
        this.currentSessionId = null;
        this.currentDataPreview = null;
        this.apiBaseUrl = '/api/v1';
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkApiConnection();
    }

    initializeElements() {
        // File upload elements
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.uploadArea = document.getElementById('uploadArea');
        this.uploadSection = document.getElementById('uploadSection');

        // Preview elements
        this.previewSection = document.getElementById('previewSection');
        this.datasetInfo = document.getElementById('datasetInfo');
        this.tabContent = document.getElementById('tabContent');
        
        // Chat elements
        this.chatSection = document.getElementById('chatSection');
        this.messages = document.getElementById('messages');
        this.queryInput = document.getElementById('queryInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.charCount = document.getElementById('charCount');

        // UI elements
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
    }

    attachEventListeners() {
        // File upload events
        if (this.uploadBtn) {
            this.uploadBtn.addEventListener('click', () => {
                if (this.fileInput) this.fileInput.click();
            });
        }
        
        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
        
        // Drag and drop events
        if (this.uploadArea) {
            this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
            this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
            this.uploadArea.addEventListener('drop', (e) => this.handleFileDrop(e));
            this.uploadArea.addEventListener('click', () => {
                if (this.fileInput) this.fileInput.click();
            });
        }

        // Preview tabs
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab-btn')) {
                this.switchTab(e.target.dataset.tab);
            }
        });

        // Chat events
        if (this.queryInput) {
            this.queryInput.addEventListener('input', () => this.updateCharCount());
            this.queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendQuery();
                }
            });
        }
        
        if (this.sendBtn) {
            this.sendBtn.addEventListener('click', () => this.sendQuery());
        }

        // Modal events
        const closeErrorModal = document.getElementById('closeErrorModal');
        const dismissError = document.getElementById('dismissError');
        
        if (closeErrorModal) closeErrorModal.addEventListener('click', () => this.hideError());
        if (dismissError) dismissError.addEventListener('click', () => this.hideError());
    }

    async checkApiConnection() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                this.updateStatus('Ready', 'success');
            } else {
                this.updateStatus('API Error', 'error');
            }
        } catch (error) {
            this.updateStatus('Offline', 'error');
            console.error('API connection failed:', error);
        }
    }

    updateStatus(text, type = 'success') {
        if (!this.statusIndicator) return;
        
        const colors = {
            success: '#22c55e',
            error: '#ef4444',
            warning: '#f59e0b'
        };
        
        this.statusIndicator.innerHTML = `<i class="fas fa-circle"></i> ${text}`;
        this.statusIndicator.style.color = colors[type] || colors.success;
    }

    // File Upload Handlers
    handleDragOver(e) {
        e.preventDefault();
        if (this.uploadArea) this.uploadArea.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        if (this.uploadArea) this.uploadArea.classList.remove('drag-over');
    }

    handleFileDrop(e) {
        e.preventDefault();
        if (this.uploadArea) this.uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        // Validate file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showError('Invalid file type. Please select a CSV file.');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File too large. Maximum size is 16MB.');
            return;
        }

        this.showLoading('Uploading and processing your CSV file...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            console.log('Sending request to:', `${this.apiBaseUrl}/session/start`);

            const response = await fetch(`${this.apiBaseUrl}/session/start`, {
                method: 'POST',
                body: formData
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error response:', errorText);
                
                let errorMessage = `HTTP ${response.status}`;
                try {
                    const errorData = JSON.parse(errorText);
                    errorMessage = errorData.message || errorData.detail || errorMessage;
                } catch (e) {
                    errorMessage = errorText || errorMessage;
                }
                throw new Error(errorMessage);
            }

            const result = await response.json();
            console.log('Success result:', result);
            
            this.currentSessionId = result.session_id;
            this.currentDataPreview = result.data_preview;
            
            this.hideLoading();
            this.showDataPreview();
            this.showChat();
            this.updateStatus('Session Active', 'success');

        } catch (error) {
            this.hideLoading();
            console.error('Upload error details:', error);
            this.showError(`Upload failed: ${error.message}`);
        }
    }

    showDataPreview() {
        if (!this.previewSection || !this.currentDataPreview) return;
        
        const preview = this.currentDataPreview;
        
        // Update dataset info
        if (this.datasetInfo) {
            this.datasetInfo.innerHTML = `
                <span><strong>${preview.shape[0]}</strong> rows, <strong>${preview.shape[1]}</strong> columns</span>
            `;
        }

        // Show preview section
        this.previewSection.style.display = 'block';
        
        // Load initial tab (sample data)
        this.switchTab('sample');
        
        // Scroll to preview
        this.previewSection.scrollIntoView({ behavior: 'smooth' });
    }

    switchTab(tabName) {
        // Update active tab
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeTab) activeTab.classList.add('active');

        // Load tab content
        if (!this.currentDataPreview || !this.tabContent) return;
        
        let content = '';
        const preview = this.currentDataPreview;

        switch (tabName) {
            case 'sample':
                content = this.renderSampleData(preview.sample_data, preview.columns);
                break;
            case 'overview':
                content = this.renderOverview(preview);
                break;
            case 'stats':
                content = this.renderStatistics(preview);
                break;
            case 'quality':
                content = this.renderDataQuality(preview);
                break;
            case 'columns':
                content = this.renderColumnsInfo(preview);
                break;
            case 'insights':
                content = this.renderInsights(preview);
                break;
            default:
                content = '<p>Tab content not implemented yet.</p>';
        }

        this.tabContent.innerHTML = content;
    }

    renderOverview(preview) {
        return `
            <div class="overview-content">
                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>Dataset Size</h4>
                        <div class="value">${preview.shape[0].toLocaleString()} × ${preview.shape[1]}</div>
                        <div class="label">rows × columns</div>
                    </div>
                    <div class="stat-card">
                        <h4>Memory Usage</h4>
                        <div class="value">${this.formatBytes(preview.memory_usage || 0)}</div>
                        <div class="label">total size</div>
                    </div>
                </div>
                <p><strong>Columns:</strong> ${preview.columns.join(', ')}</p>
                
                <div class="dataset-actions">
                    <button onclick="window.pyDataAssistant.showFullDatasetModal()" class="action-btn">
                        <i class="fas fa-table"></i> View Full Dataset
                    </button>
                    <button onclick="window.pyDataAssistant.exportData('csv')" class="action-btn">
                        <i class="fas fa-download"></i> Export CSV
                    </button>
                    <button onclick="window.pyDataAssistant.exportData('xlsx')" class="action-btn">
                        <i class="fas fa-download"></i> Export Excel
                    </button>
                </div>
            </div>
        `;
    }

    renderSampleData(sampleData, columns) {
        if (!sampleData || sampleData.length === 0) {
            return '<p class="text-muted">No sample data available</p>';
        }

        let html = '<div class="table-container"><table class="data-table"><thead><tr>';
        
        // Headers
        columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += '</tr></thead><tbody>';

        // Data rows
        sampleData.forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const value = row[col] !== null && row[col] !== undefined ? row[col] : '';
                html += `<td>${this.truncateText(String(value), 50)}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        return html;
    }

    renderStatistics(preview) {
        let html = '<div class="stats-container">';
        html += '<div class="stats-grid">';

        // Basic stats
        html += `
            <div class="stat-card">
                <h4>Total Rows</h4>
                <div class="value">${preview.shape[0].toLocaleString()}</div>
            </div>
            <div class="stat-card">
                <h4>Total Columns</h4>
                <div class="value">${preview.shape[1]}</div>
            </div>
        `;

        // Missing values summary
        if (preview.missing_values) {
            const totalMissing = Object.values(preview.missing_values).reduce((a, b) => a + b, 0);
            html += `
                <div class="stat-card">
                    <h4>Missing Values</h4>
                    <div class="value">${totalMissing.toLocaleString()}</div>
                </div>
            `;
        }

        html += '</div></div>';
        return html;
    }

    renderDataQuality(preview) {
        if (!preview.missing_values) {
            return '<p>No data quality information available.</p>';
        }

        let html = '<div class="quality-content"><h4>Missing Values by Column</h4>';
        html += '<div class="table-container"><table class="data-table"><thead>';
        html += '<tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>';
        html += '</thead><tbody>';

        Object.entries(preview.missing_values).forEach(([col, missing]) => {
            const pct = preview.missing_percentages ? preview.missing_percentages[col] || 0 : 0;
            html += `
                <tr>
                    <td><strong>${col}</strong></td>
                    <td>${missing}</td>
                    <td>${pct.toFixed(1)}%</td>
                </tr>
            `;
        });

        html += '</tbody></table></div></div>';
        return html;
    }

    renderColumnsInfo(preview) {
        let html = '<div class="columns-content"><h4>Column Information</h4>';
        html += '<div class="table-container"><table class="data-table"><thead>';
        html += '<tr><th>Column</th><th>Data Type</th></tr>';
        html += '</thead><tbody>';

        preview.columns.forEach(col => {
            const dtype = preview.dtypes ? preview.dtypes[col] || 'unknown' : 'unknown';
            html += `
                <tr>
                    <td><strong>${col}</strong></td>
                    <td>${dtype}</td>
                </tr>
            `;
        });

        html += '</tbody></table></div></div>';
        return html;
    }

    renderInsights(preview) {
        return `
            <div class="insights-content">
                <h4>📊 Dataset Insights</h4>
                <ul>
                    <li>Your dataset contains <strong>${preview.shape[0]} rows</strong> and <strong>${preview.shape[1]} columns</strong></li>
                    <li>Columns: ${preview.columns.join(', ')}</li>
                    <li>Upload completed successfully and data is ready for analysis</li>
                </ul>
                <p><em>Use the chat interface to ask specific questions about your data!</em></p>
            </div>
        `;
    }

    showChat() {
        if (!this.chatSection) return;
        
        this.chatSection.style.display = 'block';
        this.chatSection.scrollIntoView({ behavior: 'smooth' });
        
        if (this.queryInput) this.queryInput.focus();
    }

    async sendQuery() {
        if (!this.queryInput || !this.currentSessionId) return;
        
        const query = this.queryInput.value.trim();
        if (!query) return;

        // Add user message to chat
        this.addMessage('user', query);
        this.queryInput.value = '';
        this.updateCharCount();

        this.showLoading('Analyzing your query...');

        try {
            const response = await fetch(`${this.apiBaseUrl}/session/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSessionId,
                    query: query
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                let errorMessage = `HTTP ${response.status}`;
                try {
                    const errorData = JSON.parse(errorText);
                    errorMessage = errorData.message || errorData.detail || errorMessage;
                } catch (e) {
                    errorMessage = errorText || errorMessage;
                }
                throw new Error(errorMessage);
            }

            const result = await response.json();
            this.hideLoading();
            this.handleQueryResponse(result);

        } catch (error) {
            this.hideLoading();
            this.addMessage('assistant', `❌ Sorry, I encountered an error: ${error.message}`);
            console.error('Query error:', error);
        }
    }

    handleQueryResponse(response) {
        const { response_type, data, message } = response;
        
        // Handle different response types
        switch (response_type) {
            case 'plot':
                this.addPlotMessage(data, message);
                break;
            case 'statistics':
            case 'insight':
            case 'text':
                this.addFormattedMessage('assistant', data.interpretation || data.text || message);
                break;
            case 'error':
                this.addMessage('assistant', `❌ ${data.message || message}`);
                break;
            default:
                this.addMessage('assistant', message || 'Analysis completed');
        }
    }

    addMessage(role, content) {
        if (!this.messages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = role === 'user' ? 
            '<div class="message-avatar"><i class="fas fa-user"></i></div>' :
            '<div class="message-avatar"><i class="fas fa-robot"></i></div>';
        
        messageDiv.innerHTML = `
            ${avatar}
            <div class="message-content">
                <p>${content}</p>
            </div>
        `;

        this.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addFormattedMessage(role, content) {
        if (!this.messages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = role === 'user' ? 
            '<div class="message-avatar"><i class="fas fa-user"></i></div>' :
            '<div class="message-avatar"><i class="fas fa-robot"></i></div>';
        
        // Format the content to handle headers and bullet points
        const formattedContent = this.formatAnalysisText(content);
        
        messageDiv.innerHTML = `
            ${avatar}
            <div class="message-content analysis-content">
                ${formattedContent}
            </div>
        `;

        this.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addPlotMessage(plotData, message) {
        if (!this.messages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        const plotId = `plot-${Date.now()}`;
        const title = plotData.title || 'Data Visualization';
        
        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-chart-bar"></i></div>
            <div class="message-content">
                <div class="plot-container">
                    <div class="plot-header">
                        <h4>${title}</h4>
                        <div class="export-buttons">
                            <button onclick="window.pyDataAssistant.exportChart('${plotId}', 'png')" title="Export as PNG">
                                <i class="fas fa-download"></i> PNG
                            </button>
                            <button onclick="window.pyDataAssistant.exportChart('${plotId}', 'svg')" title="Export as SVG">
                                <i class="fas fa-download"></i> SVG
                            </button>
                            <button onclick="window.pyDataAssistant.exportChart('${plotId}', 'pdf')" title="Export as PDF">
                                <i class="fas fa-download"></i> PDF
                            </button>
                        </div>
                    </div>
                    <div id="${plotId}" class="plot-area"></div>
                    <p class="plot-message">${message}</p>
                </div>
            </div>
        `;
        
        this.messages.appendChild(messageDiv);
        
        // Render the Plotly chart
        if (plotData.data && typeof Plotly !== 'undefined') {
            Plotly.newPlot(plotId, plotData.data.data, plotData.data.layout, plotData.data.config);
        } else {
            // Fallback if Plotly is not available
            document.getElementById(plotId).innerHTML = '<p>⚠️ Chart data received but Plotly.js not loaded. Please refresh the page.</p>';
        }
        
        this.scrollToBottom();
    }

    updateCharCount() {
        if (!this.queryInput || !this.charCount) return;
        
        const count = this.queryInput.value.length;
        this.charCount.textContent = `${count}/1000`;
        
        if (this.sendBtn) {
            this.sendBtn.disabled = count === 0;
        }
    }

    scrollToBottom() {
        if (this.messages) {
            this.messages.scrollTop = this.messages.scrollHeight;
        }
    }

    // Loading and Error Handling
    showLoading(text = 'Processing...') {
        if (this.loadingText) this.loadingText.textContent = text;
        if (this.loadingOverlay) this.loadingOverlay.classList.add('show');
    }

    hideLoading() {
        if (this.loadingOverlay) this.loadingOverlay.classList.remove('show');
    }

    showError(message) {
        if (this.errorMessage) this.errorMessage.textContent = message;
        if (this.errorModal) this.errorModal.classList.add('show');
    }

    hideError() {
        if (this.errorModal) this.errorModal.classList.remove('show');
    }

    // Text formatting for analysis content
    formatAnalysisText(text) {
        if (!text) return '<p>No content available</p>';
        
        // Split into lines
        const lines = text.split('\n');
        let html = '';
        
        for (let line of lines) {
            line = line.trim();
            
            // Skip empty lines
            if (!line) {
                html += '<br>';
                continue;
            }
            
            // Headers (### format)
            if (line.startsWith('###') && line.endsWith('###')) {
                const headerText = line.replace(/###/g, '').trim();
                html += `<h3 class="analysis-header">${headerText}</h3>`;
                continue;
            }
            
            // Section separators (---)
            if (line.startsWith('---')) {
                const sectionText = line.replace(/---/g, '').trim();
                if (sectionText) {
                    html += `<h4 class="analysis-section">${sectionText}</h4>`;
                } else {
                    html += '<hr class="analysis-divider">';
                }
                continue;
            }
            
            // Bullet points (• or -)
            if (line.startsWith('•') || line.startsWith('-') || line.startsWith('*')) {
                const bulletText = line.replace(/^[•\-\*]\s*/, '');
                html += `<div class="analysis-bullet">• ${bulletText}</div>`;
                continue;
            }
            
            // Bold text (**text**)
            line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
            // Regular paragraph
            html += `<p class="analysis-text">${line}</p>`;
        }
        
        return html;
    }

    // Export Functions
    async exportChart(plotId, format) {
        if (!plotId || typeof Plotly === 'undefined') {
            this.showError('Unable to export chart. Plotly not available.');
            return;
        }

        try {
            const plotElement = document.getElementById(plotId);
            if (!plotElement) {
                this.showError('Chart not found.');
                return;
            }

            let filename = `chart_${Date.now()}.${format}`;
            
            if (format === 'png') {
                const imageData = await Plotly.toImage(plotElement, {
                    format: 'png',
                    width: 1200,
                    height: 800
                });
                this.downloadBase64(imageData, filename);
            } else if (format === 'svg') {
                const svgData = await Plotly.toImage(plotElement, {
                    format: 'svg',
                    width: 1200,
                    height: 800
                });
                this.downloadSVG(svgData, filename);
            } else if (format === 'pdf') {
                // For PDF, we'll convert PNG to PDF using canvas
                const imageData = await Plotly.toImage(plotElement, {
                    format: 'png',
                    width: 1200,
                    height: 800
                });
                this.downloadPDF(imageData, filename.replace('.pdf', '.png'));
            }

        } catch (error) {
            console.error('Export error:', error);
            this.showError(`Export failed: ${error.message}`);
        }
    }

    downloadBase64(base64Data, filename) {
        const link = document.createElement('a');
        link.href = base64Data;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    downloadSVG(svgData, filename) {
        const blob = new Blob([svgData], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    downloadPDF(imageData, filename) {
        // Simple PDF creation - just embed the image
        // For a full PDF solution, you'd want to use jsPDF library
        this.downloadBase64(imageData, filename);
    }

    async exportData(format = 'csv') {
        if (!this.currentSessionId) {
            this.showError('No active session. Please upload a dataset first.');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/session/export`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSessionId,
                    format: format
                })
            });

            if (!response.ok) {
                throw new Error(`Export failed: ${response.statusText}`);
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `dataset_${Date.now()}.${format}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);

        } catch (error) {
            console.error('Data export error:', error);
            this.showError(`Data export failed: ${error.message}`);
        }
    }

    // Full Dataset Modal Functions
    showFullDatasetModal() {
        if (!this.currentSessionId) {
            this.showError('No active session. Please upload a dataset first.');
            return;
        }

        // Create modal if it doesn't exist
        let modal = document.getElementById('fullDatasetModal');
        if (!modal) {
            this.createFullDatasetModal();
            modal = document.getElementById('fullDatasetModal');
        }

        // Reset pagination state
        this.datasetPagination = {
            currentPage: 1,
            pageSize: 50,
            totalRows: this.currentDataPreview?.shape[0] || 0,
            searchTerm: '',
            sortColumn: null,
            sortDirection: 'asc'
        };

        this.loadDatasetPage();
        modal.style.display = 'flex';
    }

    createFullDatasetModal() {
        const modal = document.createElement('div');
        modal.id = 'fullDatasetModal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content large">
                <div class="modal-header">
                    <h3>Full Dataset Viewer</h3>
                    <div class="dataset-controls">
                        <input type="text" id="datasetSearch" placeholder="Search..." class="search-input">
                        <select id="pageSizeSelect" class="page-size-select">
                            <option value="25">25 rows</option>
                            <option value="50" selected>50 rows</option>
                            <option value="100">100 rows</option>
                            <option value="200">200 rows</option>
                        </select>
                        <button class="close-modal" onclick="window.pyDataAssistant.closeFullDatasetModal()">×</button>
                    </div>
                </div>
                <div class="modal-body">
                    <div id="datasetTableContainer" class="dataset-table-container">
                        <div class="loading-spinner">Loading...</div>
                    </div>
                </div>
                <div class="modal-footer">
                    <div class="pagination-info" id="paginationInfo"></div>
                    <div class="pagination-controls" id="paginationControls"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Add event listeners
        const searchInput = modal.querySelector('#datasetSearch');
        const pageSizeSelect = modal.querySelector('#pageSizeSelect');
        
        searchInput.addEventListener('input', () => {
            clearTimeout(this.searchTimeout);
            this.searchTimeout = setTimeout(() => {
                this.datasetPagination.searchTerm = searchInput.value;
                this.datasetPagination.currentPage = 1;
                this.loadDatasetPage();
            }, 300);
        });

        pageSizeSelect.addEventListener('change', () => {
            this.datasetPagination.pageSize = parseInt(pageSizeSelect.value);
            this.datasetPagination.currentPage = 1;
            this.loadDatasetPage();
        });
    }

    async loadDatasetPage() {
        const container = document.getElementById('datasetTableContainer');
        if (!container) return;

        container.innerHTML = '<div class="loading-spinner">Loading...</div>';

        try {
            const response = await fetch(`${this.apiBaseUrl}/session/data`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: this.currentSessionId,
                    page: this.datasetPagination.currentPage,
                    page_size: this.datasetPagination.pageSize,
                    search: this.datasetPagination.searchTerm,
                    sort_column: this.datasetPagination.sortColumn,
                    sort_direction: this.datasetPagination.sortDirection
                })
            });

            if (!response.ok) {
                throw new Error('Failed to load dataset page');
            }

            const data = await response.json();
            this.renderDatasetTable(data);
            this.updatePaginationControls(data);

        } catch (error) {
            console.error('Load dataset page error:', error);
            container.innerHTML = '<p class="error">Failed to load data</p>';
        }
    }

    renderDatasetTable(data) {
        const container = document.getElementById('datasetTableContainer');
        if (!container) return;

        let html = '<table class="dataset-table"><thead><tr>';
        
        // Headers with sorting
        data.columns.forEach(col => {
            const sortClass = this.datasetPagination.sortColumn === col ? 
                `sorted-${this.datasetPagination.sortDirection}` : '';
            html += `
                <th class="sortable ${sortClass}" onclick="window.pyDataAssistant.sortDatasetBy('${col}')">
                    ${col}
                    <i class="fas fa-sort"></i>
                </th>
            `;
        });
        html += '</tr></thead><tbody>';

        // Data rows
        data.rows.forEach((row, index) => {
            html += '<tr>';
            data.columns.forEach(col => {
                const value = row[col] !== null && row[col] !== undefined ? row[col] : '';
                html += `<td title="${String(value)}">${this.truncateText(String(value), 100)}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    }

    updatePaginationControls(data) {
        const infoElement = document.getElementById('paginationInfo');
        const controlsElement = document.getElementById('paginationControls');
        
        if (!infoElement || !controlsElement) return;

        const totalPages = Math.ceil(data.total_rows / this.datasetPagination.pageSize);
        const startRow = (this.datasetPagination.currentPage - 1) * this.datasetPagination.pageSize + 1;
        const endRow = Math.min(startRow + data.rows.length - 1, data.total_rows);

        infoElement.innerHTML = `Showing ${startRow}-${endRow} of ${data.total_rows} rows`;

        let controlsHtml = '';
        if (totalPages > 1) {
            controlsHtml += `
                <button onclick="window.pyDataAssistant.goToDatasetPage(1)" 
                        ${this.datasetPagination.currentPage === 1 ? 'disabled' : ''}>
                    <i class="fas fa-angle-double-left"></i>
                </button>
                <button onclick="window.pyDataAssistant.goToDatasetPage(${this.datasetPagination.currentPage - 1})" 
                        ${this.datasetPagination.currentPage === 1 ? 'disabled' : ''}>
                    <i class="fas fa-angle-left"></i>
                </button>
                <span class="page-info">Page ${this.datasetPagination.currentPage} of ${totalPages}</span>
                <button onclick="window.pyDataAssistant.goToDatasetPage(${this.datasetPagination.currentPage + 1})" 
                        ${this.datasetPagination.currentPage === totalPages ? 'disabled' : ''}>
                    <i class="fas fa-angle-right"></i>
                </button>
                <button onclick="window.pyDataAssistant.goToDatasetPage(${totalPages})" 
                        ${this.datasetPagination.currentPage === totalPages ? 'disabled' : ''}>
                    <i class="fas fa-angle-double-right"></i>
                </button>
            `;
        }
        controlsElement.innerHTML = controlsHtml;
    }

    sortDatasetBy(column) {
        if (this.datasetPagination.sortColumn === column) {
            this.datasetPagination.sortDirection = this.datasetPagination.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.datasetPagination.sortColumn = column;
            this.datasetPagination.sortDirection = 'asc';
        }
        this.datasetPagination.currentPage = 1;
        this.loadDatasetPage();
    }

    goToDatasetPage(page) {
        this.datasetPagination.currentPage = page;
        this.loadDatasetPage();
    }

    closeFullDatasetModal() {
        const modal = document.getElementById('fullDatasetModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    // Utility Functions
    formatBytes(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength - 3) + '...';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing PyData Assistant...');
    window.pyDataAssistant = new PyDataAssistant();
});