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
        
        console.log('=== handleQueryResponse ===');
        console.log('Full response:', response);
        console.log('Response type:', response_type);
        console.log('Data object:', data);
        console.log('Message:', message);
        
        // Handle different response types
        switch (response_type) {
            case 'plot':
                console.log('Handling plot response');
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
                    <h4>${title}</h4>
                    <div id="${plotId}" class="plot-area"></div>
                    <p class="plot-message">${message}</p>
                </div>
            </div>
        `;
        
        this.messages.appendChild(messageDiv);
        
        // Render the Plotly chart
        if (plotData.data && typeof Plotly !== 'undefined') {
            console.log('Full plotData:', plotData);
            console.log('plotData.data:', plotData.data);
            console.log('plotData.data.data:', plotData.data.data);
            
            // plotData.data already contains the full Plotly JSON structure {data, layout, config}
            const chartData = plotData.data.data || [];
            const chartLayout = plotData.data.layout || {};
            const chartConfig = plotData.data.config || {responsive: true, displayModeBar: true};
            
            // CRITICAL FIX: Decode binary encoded data
            // Plotly's to_json() sometimes encodes numpy arrays as binary data
            if (chartData && chartData.length > 0) {
                chartData.forEach(trace => {
                    // Fix binary encoded values (common issue with numpy arrays)
                    if (trace.values && typeof trace.values === 'object' && trace.values.bdata) {
                        console.log('Decoding binary values data');
                        trace.values = this.decodeBinaryData(trace.values);
                    }
                    
                    // Fix binary encoded labels
                    if (trace.labels && typeof trace.labels === 'object' && trace.labels.bdata) {
                        console.log('Decoding binary labels data');
                        trace.labels = this.decodeBinaryData(trace.labels);
                    }
                    
                    // Fix binary encoded x/y data
                    if (trace.x && typeof trace.x === 'object' && trace.x.bdata) {
                        trace.x = this.decodeBinaryData(trace.x);
                    }
                    if (trace.y && typeof trace.y === 'object' && trace.y.bdata) {
                        trace.y = this.decodeBinaryData(trace.y);
                    }
                    
                    // Remove problematic hovertemplate if it references customdata
                    if (trace.hovertemplate && trace.hovertemplate.includes('customdata')) {
                        console.log('Removing problematic hovertemplate:', trace.hovertemplate);
                        delete trace.hovertemplate;
                    }
                    
                    // For pie charts, log trace data for debugging
                    if (trace.type === 'pie') {
                        console.log('Pie chart trace after decoding:', {
                            labels: Array.isArray(trace.labels) ? trace.labels.slice(0, 5) : trace.labels,
                            values: Array.isArray(trace.values) ? trace.values.slice(0, 5) : trace.values,
                            labelsType: typeof trace.labels,
                            valuesType: typeof trace.values
                        });
                    }
                });
            }
            
            console.log('Rendering chart with:', {chartData, chartLayout, chartConfig});
            
            Plotly.newPlot(plotId, chartData, chartLayout, chartConfig);
        } else {
            console.error('Plotly not available or no data:', {plotData, hasPlotly: typeof Plotly !== 'undefined'});
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

    // Utility Functions
    decodeBinaryData(binaryObj) {
        /**
         * Decode Plotly's binary encoded data format
         * Plotly's to_json() encodes numpy arrays as {dtype: 'f8', bdata: 'base64...'}
         */
        if (!binaryObj || !binaryObj.bdata) {
            return binaryObj;
        }
        
        try {
            const dtype = binaryObj.dtype;
            const base64Data = binaryObj.bdata;
            
            // Decode base64 to binary
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            // Convert based on dtype
            let result = [];
            if (dtype === 'f8') {
                // Float64 (8 bytes per value)
                const view = new DataView(bytes.buffer);
                for (let i = 0; i < bytes.length; i += 8) {
                    result.push(view.getFloat64(i, true)); // true = little-endian
                }
            } else if (dtype === 'f4') {
                // Float32 (4 bytes per value)
                const view = new DataView(bytes.buffer);
                for (let i = 0; i < bytes.length; i += 4) {
                    result.push(view.getFloat32(i, true));
                }
            } else if (dtype === 'i4') {
                // Int32 (4 bytes per value)
                const view = new DataView(bytes.buffer);
                for (let i = 0; i < bytes.length; i += 4) {
                    result.push(view.getInt32(i, true));
                }
            } else {
                console.warn('Unknown dtype:', dtype, '- returning original');
                return binaryObj;
            }
            
            console.log(`Decoded ${result.length} values from binary data (dtype: ${dtype})`);
            return result;
        } catch (error) {
            console.error('Error decoding binary data:', error);
            return binaryObj;
        }
    }
    
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