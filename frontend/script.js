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

        // Settings button
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsModal = document.getElementById('settingsModal');
        const closeSettingsModal = document.getElementById('closeSettingsModal');
        const saveSettingsBtn = document.getElementById('saveSettingsBtn');
        const resetSettingsBtn = document.getElementById('resetSettingsBtn');
        const clearAllDataBtn = document.getElementById('clearAllDataBtn');
        
        if (settingsBtn) settingsBtn.addEventListener('click', () => this.openSettings());
        if (closeSettingsModal) closeSettingsModal.addEventListener('click', () => this.closeSettings());
        if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        if (resetSettingsBtn) resetSettingsBtn.addEventListener('click', () => this.resetSettings());
        if (clearAllDataBtn) clearAllDataBtn.addEventListener('click', () => this.clearAllData());

        // Help button
        const helpBtn = document.getElementById('helpBtn');
        const helpModal = document.getElementById('helpModal');
        const closeHelpModal = document.getElementById('closeHelpModal');
        
        if (helpBtn) helpBtn.addEventListener('click', () => this.openHelp());
        if (closeHelpModal) closeHelpModal.addEventListener('click', () => this.closeHelp());

        // Export button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) exportBtn.addEventListener('click', () => this.exportData());
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
        
        // Show export button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) exportBtn.style.display = 'inline-block';
        
        // Load initial tab (sample data)
        this.switchTab('sample');
        
        // Show suggested queries in chat
        this.showSuggestedQueries(preview);
        
        // Scroll to preview
        this.previewSection.scrollIntoView({ behavior: 'smooth' });
    }

    showSuggestedQueries(preview) {
        if (!this.messages) return;
        
        const suggestions = this.generateSmartSuggestions(preview);
        
        if (suggestions.length === 0) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';
        
        let suggestionsHTML = `
            <div class="message-avatar"><i class="fas fa-lightbulb"></i></div>
            <div class="message-content">
                <div class="suggested-queries">
                    <h4><i class="fas fa-magic"></i> Suggested Queries</h4>
                    <p style="margin-bottom: 1rem; color: var(--text-secondary);">Click any suggestion to run it automatically:</p>
                    <div class="suggestions-grid">
        `;
        
        suggestions.forEach(suggestion => {
            suggestionsHTML += `
                <button class="suggestion-btn" onclick="window.pyDataAssistant.executeSuggestion('${suggestion.query.replace(/'/g, "\\'")}')">
                    <i class="${suggestion.icon}"></i>
                    <span>${suggestion.label}</span>
                </button>
            `;
        });
        
        suggestionsHTML += `
                    </div>
                </div>
            </div>
        `;
        
        messageDiv.innerHTML = suggestionsHTML;
        this.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    generateSmartSuggestions(preview) {
        const suggestions = [];
        const columns = preview.columns || [];
        const dtypes = preview.dtypes || {};
        
        // Get numeric and categorical columns
        const numericCols = columns.filter(col => 
            dtypes[col] && (dtypes[col].includes('int') || dtypes[col].includes('float'))
        );
        const categoricalCols = columns.filter(col => 
            dtypes[col] && dtypes[col].includes('object')
        );
        
        // Check for date/time columns
        const dateCols = columns.filter(col => 
            dtypes[col] && (dtypes[col].includes('datetime') || col.toLowerCase().includes('date') || col.toLowerCase().includes('time'))
        );
        
        // Data overview
        suggestions.push({
            icon: 'fas fa-eye',
            label: 'Show dataset overview and summary',
            query: 'Give me a comprehensive overview of this dataset including shape, columns, data types, and basic statistics'
        });
        
        // Missing values if any exist
        const hasMissing = preview.missing_values && 
            Object.values(preview.missing_values).some(v => v > 0);
        if (hasMissing) {
            suggestions.push({
                icon: 'fas fa-exclamation-triangle',
                label: 'Analyze missing values',
                query: 'Show me a detailed analysis of missing values in this dataset with visualizations'
            });
        }
        
        // Distribution visualizations for numeric columns
        if (numericCols.length > 0) {
            const firstNumeric = numericCols[0];
            suggestions.push({
                icon: 'fas fa-chart-bar',
                label: `Distribution of ${firstNumeric}`,
                query: `Create a histogram showing the distribution of ${firstNumeric}`
            });
            
            // Box plot for outlier detection
            if (categoricalCols.length > 0) {
                suggestions.push({
                    icon: 'fas fa-box',
                    label: `${firstNumeric} box plot by ${categoricalCols[0]}`,
                    query: `Create a box plot of ${firstNumeric} grouped by ${categoricalCols[0]} to show distribution and outliers`
                });
            }
        }
        
        // Multi-dimensional visualizations
        if (numericCols.length >= 2) {
            suggestions.push({
                icon: 'fas fa-project-diagram',
                label: 'Correlation heatmap',
                query: 'Show me correlations between numeric columns with a heatmap'
            });
            
            suggestions.push({
                icon: 'fas fa-chart-scatter',
                label: `${numericCols[0]} vs ${numericCols[1]}`,
                query: `Create a scatter plot comparing ${numericCols[0]} and ${numericCols[1]}`
            });
            
            // Bubble chart if we have 3+ numeric columns
            if (numericCols.length >= 3) {
                suggestions.push({
                    icon: 'fas fa-circle',
                    label: `Bubble chart (3D view)`,
                    query: `Create a bubble chart with ${numericCols[0]} on x-axis, ${numericCols[1]} on y-axis, and ${numericCols[2]} as bubble size`
                });
            }
        }
        
        // Time series analysis
        if (dateCols.length > 0 && numericCols.length > 0) {
            suggestions.push({
                icon: 'fas fa-chart-line',
                label: `Trend over time`,
                query: `Create a line chart showing ${numericCols[0]} over ${dateCols[0]}`
            });
            
            suggestions.push({
                icon: 'fas fa-chart-area',
                label: `Area chart of trends`,
                query: `Create an area chart showing cumulative ${numericCols[0]} over ${dateCols[0]}`
            });
        }
        
        // Categorical analysis with diverse chart types
        if (categoricalCols.length > 0 && numericCols.length > 0) {
            const cat = categoricalCols[0];
            const num = numericCols[0];
            
            suggestions.push({
                icon: 'fas fa-chart-pie',
                label: `${cat} distribution`,
                query: `Create a pie chart showing the distribution of ${cat}`
            });
            
            suggestions.push({
                icon: 'fas fa-chart-column',
                label: `${num} by ${cat}`,
                query: `Show me a bar chart of total ${num} grouped by ${cat}`
            });
            
            // Violin plot for distribution comparison
            suggestions.push({
                icon: 'fas fa-music',
                label: `${num} violin plot by ${cat}`,
                query: `Create a violin plot comparing distribution of ${num} across different ${cat} categories`
            });
        }
        
        // Hierarchical visualizations
        if (categoricalCols.length >= 2 && numericCols.length > 0) {
            suggestions.push({
                icon: 'fas fa-sun',
                label: `Hierarchical sunburst`,
                query: `Create a sunburst chart showing ${numericCols[0]} broken down by ${categoricalCols[0]} and ${categoricalCols[1]}`
            });
            
            suggestions.push({
                icon: 'fas fa-th-large',
                label: `Treemap visualization`,
                query: `Create a treemap of ${numericCols[0]} by ${categoricalCols[0]} and ${categoricalCols[1]}`
            });
        }
        
        // Statistical analysis
        if (numericCols.length > 0) {
            suggestions.push({
                icon: 'fas fa-calculator',
                label: 'Statistical summary',
                query: 'Provide detailed statistical summaries for all numeric columns'
            });
        }
        
        // Outlier detection
        if (numericCols.length > 0) {
            suggestions.push({
                icon: 'fas fa-search',
                label: 'Detect outliers',
                query: 'Identify and visualize outliers in the numeric columns'
            });
        }
        
        // Advanced Statistical Analysis Suite
        if (numericCols.length > 0) {
            // Normality tests
            suggestions.push({
                icon: 'fas fa-chart-bell',
                label: 'Test normality of data',
                query: `Test if ${numericCols[0]} follows a normal distribution using Shapiro-Wilk, D'Agostino-Pearson, and Anderson-Darling tests`
            });
            
            // Distribution fitting
            suggestions.push({
                icon: 'fas fa-wave-square',
                label: 'Fit distributions',
                query: `Fit various probability distributions (normal, exponential, gamma, etc.) to ${numericCols[0]} and find the best fit`
            });
        }
        
        // Hypothesis testing
        if (numericCols.length >= 2) {
            suggestions.push({
                icon: 'fas fa-balance-scale',
                label: 'Compare two groups (T-test)',
                query: `Perform independent t-test to compare ${numericCols[0]} and ${numericCols[1]}, including effect size (Cohen's d)`
            });
            
            suggestions.push({
                icon: 'fas fa-link',
                label: 'Test correlation significance',
                query: `Test correlation between ${numericCols[0]} and ${numericCols[1]} using Pearson, Spearman, and Kendall methods`
            });
        }
        
        // Advanced outlier detection
        if (numericCols.length > 0) {
            suggestions.push({
                icon: 'fas fa-bullseye',
                label: 'Advanced outlier detection',
                query: `Detect outliers in ${numericCols[0]} using multiple methods: IQR, Z-score, Modified Z-score (MAD), and Isolation Forest`
            });
        }
        
        // Regression analysis
        if (numericCols.length >= 2) {
            suggestions.push({
                icon: 'fas fa-chart-line-up',
                label: 'Regression analysis',
                query: `Perform linear regression with ${numericCols[0]} as predictor and ${numericCols[1]} as target, including R², RMSE, and residual analysis`
            });
            
            if (numericCols.length >= 3) {
                suggestions.push({
                    icon: 'fas fa-bezier-curve',
                    label: 'Polynomial regression',
                    query: `Perform polynomial regression to model non-linear relationship between ${numericCols[0]} and ${numericCols[1]}`
                });
            }
        }
        
        // ANOVA for multiple groups
        if (categoricalCols.length > 0 && numericCols.length > 0) {
            suggestions.push({
                icon: 'fas fa-layer-group',
                label: 'Compare multiple groups (ANOVA)',
                query: `Perform one-way ANOVA to compare ${numericCols[0]} across different ${categoricalCols[0]} groups, including eta-squared effect size`
            });
            
            suggestions.push({
                icon: 'fas fa-table-cells',
                label: 'Chi-square test',
                query: `Perform chi-square test of independence between ${categoricalCols[0]} and ${categoricalCols.length > 1 ? categoricalCols[1] : numericCols[0]}, including Cramér's V`
            });
        }
        
        // Time series analysis
        if (dateCols.length > 0 && numericCols.length > 0) {
            suggestions.push({
                icon: 'fas fa-clock',
                label: 'Time series stationarity test',
                query: `Test if ${numericCols[0]} time series is stationary using Augmented Dickey-Fuller test`
            });
            
            if (numericCols.length >= 2) {
                suggestions.push({
                    icon: 'fas fa-arrows-alt-h',
                    label: 'Granger causality test',
                    query: `Test if ${numericCols[0]} Granger-causes ${numericCols[1]} (predictive causality in time series)`
                });
            }
        }
        
        return suggestions.slice(0, 12); // Increased limit to show more statistical options
    }

    executeSuggestion(query) {
        if (!this.queryInput) return;
        
        this.queryInput.value = query;
        this.updateCharCount();
        this.sendQuery();
    }

    executeStatisticalQuery(query) {
        // Same as executeSuggestion but with explicit focus on chat
        if (!this.queryInput) return;
        
        // Scroll to chat section
        const chatSection = document.getElementById('chatSection');
        if (chatSection) {
            chatSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        
        // Set query and send
        this.queryInput.value = query;
        this.updateCharCount();
        
        // Small delay to ensure smooth scroll
        setTimeout(() => {
            this.sendQuery();
        }, 300);
    }

    runCustomStatisticalPrompt() {
        // Get custom prompt from textarea
        const promptTextarea = document.getElementById('customStatsPrompt');
        if (!promptTextarea) {
            this.showError('Custom prompt input not found');
            return;
        }
        
        const userPrompt = promptTextarea.value.trim();
        if (!userPrompt) {
            this.showError('Please enter a statistical query first');
            return;
        }
        
        // Get available columns for context enhancement
        const preview = this.currentDataPreview;
        if (!preview) {
            this.showError('No data loaded');
            return;
        }
        
        const columns = preview.columns || [];
        const dtypes = preview.dtypes || {};
        
        const numericCols = columns.filter(col => 
            dtypes[col] && (dtypes[col].includes('int') || dtypes[col].includes('float'))
        );
        const categoricalCols = columns.filter(col => 
            dtypes[col] && dtypes[col].includes('object')
        );
        
        // Enhance the prompt with statistical context
        let enhancedPrompt = userPrompt;
        
        // Add column context if not already mentioned
        const mentionsColumns = columns.some(col => 
            userPrompt.toLowerCase().includes(col.toLowerCase())
        );
        
        if (!mentionsColumns && numericCols.length > 0) {
            enhancedPrompt += `\n\nContext: Dataset has numeric columns: ${numericCols.join(', ')}`;
            if (categoricalCols.length > 0) {
                enhancedPrompt += ` and categorical columns: ${categoricalCols.join(', ')}`;
            }
        }
        
        // Add statistical method hints based on keywords
        const keywords = userPrompt.toLowerCase();
        
        if (keywords.includes('normal') || keywords.includes('distribution')) {
            enhancedPrompt += '\n\n📊 Use Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov normality tests with p-values and test statistics.';
        }
        
        if (keywords.includes('outlier')) {
            enhancedPrompt += '\n\n📊 Use multiple outlier detection methods: IQR, Z-score, Modified Z-score (MAD), and Isolation Forest.';
        }
        
        if (keywords.includes('correlation') || keywords.includes('correlate')) {
            enhancedPrompt += '\n\n📊 Calculate Pearson, Spearman, and Kendall correlation coefficients with p-values.';
        }
        
        if (keywords.includes('regression') || keywords.includes('predict')) {
            enhancedPrompt += '\n\n📊 Include R², RMSE, coefficients, p-values, and residual analysis. Remove rows with missing values using .dropna() before fitting.';
        }
        
        if (keywords.includes('test') && (keywords.includes('difference') || keywords.includes('compare'))) {
            enhancedPrompt += '\n\n📊 Use appropriate hypothesis tests (t-test, ANOVA, chi-square) with effect sizes and interpretation.';
        }
        
        // Add instruction for statistical libraries
        enhancedPrompt += '\n\n⚠️ Use pre-imported statistical libraries (scipy.stats, sklearn, statsmodels) - DO NOT add import statements.';
        
        console.log('Original prompt:', userPrompt);
        console.log('Enhanced prompt:', enhancedPrompt);
        
        // Execute the enhanced query
        this.executeStatisticalQuery(enhancedPrompt);
        
        // Clear the textarea
        promptTextarea.value = '';
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
            case 'fulldata':
                this.renderFullDataTab();
                return; // Handle separately with async loading
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
            case 'statistical':
                content = this.renderStatisticalAnalysisTab(preview);
                break;
            case 'chartgallery':
                content = this.renderChartGallery();
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

    async renderFullDataTab() {
        if (!this.currentSessionId) return;
        
        this.tabContent.innerHTML = '<div class="loading-spinner"><i class="fas fa-spinner fa-spin"></i> Loading full dataset...</div>';
        
        try {
            await this.loadFullData(1);
        } catch (error) {
            this.tabContent.innerHTML = `<p class="error-text">Error loading data: ${error.message}</p>`;
        }
    }

    async loadFullData(page = 1, pageSize = 100) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/session/${this.currentSessionId}/data?page=${page}&page_size=${pageSize}`);
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.message || 'Failed to load data');
            }
            
            this.renderFullDataTable(result, page, pageSize);
        } catch (error) {
            console.error('Error loading full data:', error);
            this.tabContent.innerHTML = `<p class="error-text">❌ Error: ${error.message}</p>`;
        }
    }

    renderFullDataTable(result, currentPage, pageSize) {
        const { data, columns, pagination } = result;
        
        let html = '<div class="full-data-container">';
        
        // Pagination info
        html += `
            <div class="pagination-header">
                <div class="pagination-info">
                    Showing ${pagination.start_row}-${pagination.end_row} of ${pagination.total_rows.toLocaleString()} rows
                </div>
                <div class="pagination-controls">
                    <button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="window.pyDataAssistant.loadFullData(1, ${pageSize})">
                        <i class="fas fa-angle-double-left"></i>
                    </button>
                    <button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="window.pyDataAssistant.loadFullData(${currentPage - 1}, ${pageSize})">
                        <i class="fas fa-angle-left"></i> Prev
                    </button>
                    <span class="page-number">Page ${currentPage} of ${pagination.total_pages}</span>
                    <button class="page-btn" ${currentPage === pagination.total_pages ? 'disabled' : ''} onclick="window.pyDataAssistant.loadFullData(${currentPage + 1}, ${pageSize})">
                        Next <i class="fas fa-angle-right"></i>
                    </button>
                    <button class="page-btn" ${currentPage === pagination.total_pages ? 'disabled' : ''} onclick="window.pyDataAssistant.loadFullData(${pagination.total_pages}, ${pageSize})">
                        <i class="fas fa-angle-double-right"></i>
                    </button>
                </div>
                <div class="page-size-selector">
                    <label>Rows per page:</label>
                    <select onchange="window.pyDataAssistant.loadFullData(1, parseInt(this.value))">
                        <option value="50" ${pageSize === 50 ? 'selected' : ''}>50</option>
                        <option value="100" ${pageSize === 100 ? 'selected' : ''}>100</option>
                        <option value="200" ${pageSize === 200 ? 'selected' : ''}>200</option>
                        <option value="500" ${pageSize === 500 ? 'selected' : ''}>500</option>
                    </select>
                </div>
            </div>
        `;
        
        // Data table
        html += '<div class="table-container scrollable-table"><table class="data-table full-data-table"><thead><tr>';
        html += '<th class="row-index">#</th>';
        columns.forEach(col => {
            html += `<th>${col}</th>`;
        });
        html += '</tr></thead><tbody>';
        
        data.forEach((row, idx) => {
            const rowNum = pagination.start_row + idx;
            html += `<tr><td class="row-index">${rowNum}</td>`;
            columns.forEach(col => {
                const value = row[col] !== null && row[col] !== undefined ? row[col] : '';
                html += `<td>${this.truncateText(String(value), 100)}</td>`;
            });
            html += '</tr>';
        });
        
        html += '</tbody></table></div>';
        html += '</div>';
        
        this.tabContent.innerHTML = html;
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

    renderStatisticalAnalysisTab(preview) {
        const columns = preview.columns || [];
        const dtypes = preview.dtypes || {};
        
        // Get numeric and categorical columns
        const numericCols = columns.filter(col => 
            dtypes[col] && (dtypes[col].includes('int') || dtypes[col].includes('float'))
        );
        const categoricalCols = columns.filter(col => 
            dtypes[col] && dtypes[col].includes('object')
        );

        let html = '<div class="statistical-analysis-content">';
        html += '<h3><i class="fas fa-flask"></i> Professional Statistical Analysis Suite</h3>';
        html += '<p class="stats-intro">🎉 <strong>11 Advanced Statistical Tools</strong> - Click any analysis below to run it on your data!</p>';
        
        // Custom Prompt Input Section (TOP OF PAGE)
        html += '<div class="custom-stats-prompt-section">';
        html += '<h4><i class="fas fa-keyboard"></i> Custom Statistical Query</h4>';
        html += '<p class="prompt-description">Type your own analysis request and we\'ll automatically enhance it with statistical context:</p>';
        html += '<div class="custom-prompt-container">';
        html += '<textarea id="customStatsPrompt" class="custom-stats-textarea" placeholder="Example: Test if my data is normally distributed\nExample: Find correlations between all numeric columns\nExample: Detect outliers in my dataset\nExample: Perform regression analysis on Sales vs Profit" rows="3"></textarea>';
        html += '<div class="prompt-actions">';
        html += '<button class="stat-btn-primary" onclick="window.pyDataAssistant.runCustomStatisticalPrompt()"><i class="fas fa-magic"></i> Enhance & Run Analysis</button>';
        html += '<button class="stat-btn-secondary" onclick="document.getElementById(\'customStatsPrompt\').value=\'\'"><i class="fas fa-eraser"></i> Clear</button>';
        html += '</div>';
        html += '<div class="prompt-hints">';
        html += '<small><i class="fas fa-info-circle"></i> <strong>Available columns:</strong> ';
        if (numericCols.length > 0) html += `<span class="hint-badge">Numeric: ${numericCols.slice(0, 3).join(', ')}${numericCols.length > 3 ? '...' : ''}</span>`;
        if (categoricalCols.length > 0) html += ` <span class="hint-badge">Categorical: ${categoricalCols.slice(0, 3).join(', ')}${categoricalCols.length > 3 ? '...' : ''}</span>`;
        html += '</small>';
        html += '</div>';
        html += '</div>';
        html += '</div>';
        
        // Hypothesis Testing
        html += '<div class="stats-category">';
        html += '<h4><i class="fas fa-vial"></i> Hypothesis Testing</h4>';
        html += '<div class="stats-grid">';
        
        // Normality Tests
        if (numericCols.length > 0) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-chart-bell"></i></div>
                    <h5>Normality Tests</h5>
                    <p>Test if data follows normal distribution</p>
                    <div class="tool-actions" style="display: flex; gap: 8px;">
                        <button class="stat-btn" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Test if ${numericCols[0]} is normally distributed using Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov tests. Show test statistics and p-values in TEXT FORMAT ONLY')">
                            <i class="fas fa-list"></i> Analyze
                        </button>
                        <button class="stat-btn stat-btn-viz" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Create a histogram with normal distribution overlay for ${numericCols[0]} to visualize normality')">
                            <i class="fas fa-chart-histogram"></i> Visualize
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Analyze:</strong> Shapiro-Wilk, Anderson-Darling, KS | <strong>Visualize:</strong> Histogram + normal curve</small>
                    </div>
                </div>
            `;
        }

        // T-Tests
        if (numericCols.length >= 2) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-balance-scale"></i></div>
                    <h5>T-Test (Compare Groups)</h5>
                    <p>Compare means between two groups with effect size</p>
                    <div class="tool-actions">
                        <button class="stat-btn" onclick="window.pyDataAssistant.executeStatisticalQuery('Perform independent t-test comparing ${numericCols[0]} and ${numericCols[1]} including Cohen\\'s d effect size')">
                            Compare ${numericCols[0]} vs ${numericCols[1]}
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Includes:</strong> t-statistic, p-value, Cohen's d, confidence interval</small>
                    </div>
                </div>
            `;
        }

        // ANOVA
        if (categoricalCols.length > 0 && numericCols.length > 0) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-layer-group"></i></div>
                    <h5>ANOVA (Multiple Groups)</h5>
                    <p>Compare means across 3+ groups</p>
                    <div class="tool-actions">
                        <button class="stat-btn" onclick="window.pyDataAssistant.executeStatisticalQuery('Perform one-way ANOVA comparing ${numericCols[0]} across different ${categoricalCols[0]} groups with eta-squared effect size')">
                            ${numericCols[0]} by ${categoricalCols[0]}
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Includes:</strong> F-statistic, p-value, eta-squared</small>
                    </div>
                </div>
            `;
        }

        // Chi-Square
        if (categoricalCols.length >= 2) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-table-cells"></i></div>
                    <h5>Chi-Square Test</h5>
                    <p>Test independence between categorical variables</p>
                    <div class="tool-actions">
                        <button class="stat-btn" onclick="window.pyDataAssistant.executeStatisticalQuery('Perform chi-square test of independence between ${categoricalCols[0]} and ${categoricalCols[1]} including Cramér\\'s V')">
                            ${categoricalCols[0]} vs ${categoricalCols[1]}
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Includes:</strong> χ² statistic, p-value, Cramér's V</small>
                    </div>
                </div>
            `;
        }

        // Correlation
        if (numericCols.length >= 2) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-link"></i></div>
                    <h5>Correlation Analysis</h5>
                    <p>Test correlation significance (3 methods)</p>
                    <div class="tool-actions" style="display: flex; gap: 8px;">
                        <button class="stat-btn" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Test correlation between ${numericCols[0]} and ${numericCols[1]} using Pearson, Spearman, and Kendall methods. Show coefficients and p-values in TEXT FORMAT ONLY')">
                            <i class="fas fa-list"></i> Analyze
                        </button>
                        <button class="stat-btn stat-btn-viz" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Create a scatter plot showing correlation between ${numericCols[0]} and ${numericCols[1]} with trend line and Pearson correlation coefficient')">
                            <i class="fas fa-chart-scatter"></i> Visualize
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Analyze:</strong> Pearson, Spearman, Kendall | <strong>Visualize:</strong> Scatter plot with trend line</small>
                    </div>
                </div>
            `;
        }

        html += '</div></div>'; // Close hypothesis testing

        // Outlier Detection
        html += '<div class="stats-category">';
        html += '<h4><i class="fas fa-bullseye"></i> Outlier Detection</h4>';
        html += '<div class="stats-grid">';
        
        if (numericCols.length > 0) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-search"></i></div>
                    <h5>Advanced Outlier Detection</h5>
                    <p>Detect anomalies using 4 different methods</p>
                    <div class="tool-actions" style="display: flex; gap: 8px;">
                        <button class="stat-btn" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Detect outliers in ${numericCols[0]} using all methods: IQR, Z-score, Modified Z-score (MAD), and Isolation Forest. Show counts and threshold values in TEXT FORMAT ONLY')">
                            <i class="fas fa-list"></i> Analyze
                        </button>
                        <button class="stat-btn stat-btn-viz" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Create a box plot for ${numericCols[0]} to visualize outliers with IQR method. Highlight outliers in red')">
                            <i class="fas fa-chart-box"></i> Visualize
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Analyze:</strong> IQR, Z-score, MAD, Isolation Forest | <strong>Visualize:</strong> Box plot with outliers</small>
                    </div>
                </div>
            `;
        }

        html += '</div></div>'; // Close outlier detection

        // Regression Analysis
        html += '<div class="stats-category">';
        html += '<h4><i class="fas fa-chart-line-up"></i> Regression Analysis</h4>';
        html += '<div class="stats-grid">';
        
        if (numericCols.length >= 2) {
            // Smart variable selection: Exclude ID-like columns (Postal Code, Row ID, etc.)
            const meaningfulCols = numericCols.filter(col => 
                !col.toLowerCase().includes('id') && 
                !col.toLowerCase().includes('postal') &&
                !col.toLowerCase().includes('code') &&
                !col.toLowerCase().includes('index')
            );
            
            const predictor = meaningfulCols.length >= 2 ? meaningfulCols[0] : numericCols[0];
            const target = meaningfulCols.length >= 2 ? meaningfulCols[1] : numericCols[1];
            
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-chart-line"></i></div>
                    <h5>Linear Regression</h5>
                    <p>Model linear relationships between variables</p>
                    <div class="tool-actions" style="display: flex; gap: 8px;">
                        <button class="stat-btn" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Perform linear regression with ${predictor} as predictor and ${target} as target. Show R², RMSE, coefficients, and interpretation. IMPORTANT: If R² is below 0.3, warn that these variables have weak correlation and suggest trying different variable pairs. Show first 10 predictions in TEXT FORMAT ONLY - do not create any visualization')">
                            <i class="fas fa-list"></i> Analyze
                        </button>
                        <button class="stat-btn stat-btn-viz" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Create a scatter plot showing ${target} vs ${predictor} with linear regression line. Include regression equation and R² score. IMPORTANT: If R² is below 0.3, add a warning annotation that this is a weak correlation')">
                            <i class="fas fa-chart-scatter"></i> Visualize
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Current:</strong> ${predictor} → ${target} | <strong>Tip:</strong> Click "Find Best Pair" for strongest correlation</small>
                    </div>
                    <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border-color);">
                        <button class="stat-btn" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);" onclick="window.pyDataAssistant.executeStatisticalQuery('Find the best variable pair for linear regression by calculating correlation between all numeric columns. Show the top 3 pairs with highest absolute correlation coefficients and their R² values. Exclude ID columns, postal codes, and index columns. Show results in TEXT FORMAT ONLY')">
                            <i class="fas fa-wand-magic"></i> Find Best Pair (Auto-Select)
                        </button>
                    </div>
                </div>
            `;

            if (numericCols.length >= 3) {
                html += `
                    <div class="stat-tool-card">
                        <div class="tool-icon"><i class="fas fa-bezier-curve"></i></div>
                        <h5>Polynomial Regression</h5>
                        <p>Model non-linear relationships with curves</p>
                        <div class="tool-actions" style="display: flex; gap: 8px;">
                            <button class="stat-btn" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Perform polynomial regression (degree 2) with ${numericCols[0]} and ${numericCols[1]}. Show R², RMSE, coefficients in TEXT FORMAT ONLY - do not create any visualization')">
                                <i class="fas fa-list"></i> Analyze
                            </button>
                            <button class="stat-btn stat-btn-viz" style="flex: 1;" onclick="window.pyDataAssistant.executeStatisticalQuery('Create a scatter plot showing ${numericCols[1]} vs ${numericCols[0]} with polynomial regression curve (degree 2). Show polynomial equation and R² score')">
                                <i class="fas fa-chart-line"></i> Visualize
                            </button>
                        </div>
                        <div class="tool-info">
                            <small><strong>Analyze:</strong> Coefficients, R², RMSE | <strong>Visualize:</strong> Scatter + polynomial curve</small>
                        </div>
                    </div>
                `;
            }
        }

        html += '</div></div>'; // Close regression

        // Distribution Fitting
        html += '<div class="stats-category">';
        html += '<h4><i class="fas fa-wave-square"></i> Distribution Fitting</h4>';
        html += '<div class="stats-grid">';
        
        if (numericCols.length > 0) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-chart-area"></i></div>
                    <h5>Fit Probability Distributions</h5>
                    <p>Test which distribution best fits your data</p>
                    <div class="tool-actions">
                        <button class="stat-btn" onclick="window.pyDataAssistant.executeStatisticalQuery('Fit various probability distributions (normal, exponential, gamma, beta, lognormal, weibull, etc.) to ${numericCols[0]} and find the best fit')">
                            Fit ${numericCols[0]}
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>13 Distributions:</strong> normal, exponential, gamma, beta, lognormal, weibull, uniform, chi2, t, f, pareto, logistic, gumbel</small>
                    </div>
                </div>
            `;
        }

        html += '</div></div>'; // Close distribution fitting

        // Summary Statistics
        html += '<div class="stats-category">';
        html += '<h4><i class="fas fa-calculator"></i> Summary Statistics</h4>';
        html += '<div class="stats-grid">';
        
        if (numericCols.length > 0) {
            html += `
                <div class="stat-tool-card">
                    <div class="tool-icon"><i class="fas fa-list-ol"></i></div>
                    <h5>Comprehensive Statistics</h5>
                    <p>Get detailed descriptive statistics</p>
                    <div class="tool-actions">
                        <button class="stat-btn" onclick="window.pyDataAssistant.executeStatisticalQuery('Get comprehensive statistics for ${numericCols[0]} including mean, median, mode, std, variance, skewness, kurtosis, quartiles, and range')">
                            Analyze ${numericCols[0]}
                        </button>
                    </div>
                    <div class="tool-info">
                        <small><strong>Metrics:</strong> Central tendency, dispersion, shape, position</small>
                    </div>
                </div>
            `;
        }

        html += '</div></div>'; // Close summary stats

        // Custom Query Section
        html += '<div class="stats-category">';
        html += '<h4><i class="fas fa-keyboard"></i> Custom Analysis</h4>';
        html += '<div class="custom-stats-input">';
        html += '<p>Or type your own statistical analysis query in the chat below. Examples:</p>';
        html += '<ul class="stats-examples">';
        if (numericCols.length > 0) {
            html += `<li>Test if ${numericCols[0]} follows a normal distribution</li>`;
            html += `<li>Detect outliers in ${numericCols[0]} using Isolation Forest</li>`;
        }
        if (numericCols.length >= 2) {
            html += `<li>Perform regression with ${numericCols[0]} predicting ${numericCols[1]}</li>`;
            html += `<li>Test correlation between ${numericCols[0]} and ${numericCols[1]}</li>`;
        }
        html += '</ul>';
        html += '</div></div>';

        html += '</div>'; // Close main container
        
        // Add CSS for the statistical analysis tab
        html += `
            <style>
                .statistical-analysis-content {
                    padding: 1.5rem;
                }
                .stats-intro {
                    color: var(--primary-color);
                    margin-bottom: 2rem;
                    font-size: 1.05em;
                }
                .stats-category {
                    margin-bottom: 2.5rem;
                }
                .stats-category h4 {
                    color: var(--text-primary);
                    margin-bottom: 1rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 2px solid var(--primary-color);
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 1rem;
                }
                .stat-tool-card {
                    background: var(--surface-color);
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    padding: 1.25rem;
                    transition: all 0.3s ease;
                }
                .stat-tool-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    border-color: var(--primary-color);
                }
                .tool-icon {
                    font-size: 2rem;
                    color: var(--primary-color);
                    margin-bottom: 0.75rem;
                }
                .stat-tool-card h5 {
                    margin: 0 0 0.5rem 0;
                    color: var(--text-primary);
                }
                .stat-tool-card p {
                    color: var(--text-secondary);
                    font-size: 0.9em;
                    margin-bottom: 1rem;
                }
                .tool-actions {
                    margin-bottom: 0.75rem;
                }
                .stat-btn {
                    background: var(--primary-color);
                    color: white;
                    border: none;
                    padding: 0.6rem 1rem;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 0.9em;
                    width: 100%;
                    transition: all 0.2s ease;
                }
                .stat-btn:hover {
                    background: var(--primary-hover);
                    transform: scale(1.02);
                }
                .stat-btn-viz {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .stat-btn-viz:hover {
                    background: linear-gradient(135deg, #5568d3 0%, #63408b 100%);
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
                }
                .stat-btn i {
                    margin-right: 0.4rem;
                }
                .tool-info {
                    padding-top: 0.75rem;
                    border-top: 1px solid var(--border-color);
                }
                .tool-info small {
                    color: var(--text-secondary);
                    font-size: 0.85em;
                }
                .custom-stats-input {
                    background: var(--surface-color);
                    padding: 1.5rem;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                }
                .stats-examples {
                    margin: 0.75rem 0 0 1.5rem;
                }
                .stats-examples li {
                    color: var(--text-secondary);
                    margin: 0.5rem 0;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                }
                
                /* Custom Prompt Section Styles */
                .custom-stats-prompt-section {
                    background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
                    border: 2px solid var(--primary-color);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 2rem;
                }
                .custom-stats-prompt-section h4 {
                    color: var(--primary-color);
                    margin: 0 0 0.5rem 0;
                }
                .prompt-description {
                    color: var(--text-secondary);
                    margin-bottom: 1rem;
                    font-size: 0.95em;
                }
                .custom-prompt-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                .custom-stats-textarea {
                    width: 100%;
                    padding: 1rem;
                    border: 2px solid var(--border-color);
                    border-radius: 8px;
                    background: var(--surface-color);
                    color: var(--text-primary);
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    font-size: 0.95em;
                    resize: vertical;
                    min-height: 80px;
                    transition: all 0.3s ease;
                }
                .custom-stats-textarea:focus {
                    outline: none;
                    border-color: var(--primary-color);
                    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
                }
                .custom-stats-textarea::placeholder {
                    color: var(--text-secondary);
                    opacity: 0.6;
                }
                .prompt-actions {
                    display: flex;
                    gap: 0.75rem;
                }
                .stat-btn-primary {
                    flex: 1;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 0.8rem 1.5rem;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 1em;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
                }
                .stat-btn-primary:hover {
                    background: linear-gradient(135deg, #5568d3 0%, #63408b 100%);
                    transform: translateY(-2px);
                    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
                }
                .stat-btn-primary i {
                    margin-right: 0.5rem;
                }
                .stat-btn-secondary {
                    background: var(--surface-color);
                    color: var(--text-secondary);
                    border: 2px solid var(--border-color);
                    padding: 0.8rem 1.5rem;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 1em;
                    transition: all 0.3s ease;
                }
                .stat-btn-secondary:hover {
                    background: var(--border-color);
                    border-color: var(--text-secondary);
                }
                .stat-btn-secondary i {
                    margin-right: 0.5rem;
                }
                .prompt-hints {
                    background: var(--surface-color);
                    padding: 0.75rem;
                    border-radius: 6px;
                    border-left: 3px solid var(--primary-color);
                }
                .prompt-hints small {
                    color: var(--text-secondary);
                }
                .hint-badge {
                    display: inline-block;
                    background: var(--primary-color);
                    color: white;
                    padding: 0.2rem 0.6rem;
                    border-radius: 4px;
                    font-size: 0.85em;
                    margin: 0.2rem;
                }
            </style>
        `;

        return html;
    }

    renderChartGallery() {
        const chartTypes = [
            {
                category: 'Distribution & Comparison',
                charts: [
                    { name: 'Bar Chart', icon: 'fa-chart-bar', description: 'Compare categories or values across groups', example: 'Show me a bar chart of sales by region' },
                    { name: 'Histogram', icon: 'fa-chart-area', description: 'Visualize data distribution and frequency', example: 'Create a histogram of age distribution' },
                    { name: 'Box Plot', icon: 'fa-box', description: 'Display quartiles, median, and outliers', example: 'Show box plot of prices by category' },
                    { name: 'Violin Plot', icon: 'fa-music', description: 'Combine box plot with kernel density', example: 'Create violin plot comparing salaries across departments' }
                ]
            },
            {
                category: 'Relationships & Correlations',
                charts: [
                    { name: 'Scatter Plot', icon: 'fa-braille', description: 'Show relationship between two variables', example: 'Scatter plot of age vs income' },
                    { name: 'Bubble Chart', icon: 'fa-circle', description: 'Scatter plot with size dimension', example: 'Show population vs GDP with bubble sizes for area' },
                    { name: 'Heatmap', icon: 'fa-th', description: 'Display correlations in a color matrix', example: 'Create a correlation heatmap of all numeric columns' },
                    { name: 'Line Chart', icon: 'fa-chart-line', description: 'Show trends over time or sequences', example: 'Line chart of monthly revenue over time' }
                ]
            },
            {
                category: 'Proportions & Parts',
                charts: [
                    { name: 'Pie Chart', icon: 'fa-chart-pie', description: 'Show percentage breakdown of categories', example: 'Pie chart of market share by product' },
                    { name: 'Donut Chart', icon: 'fa-circle-notch', description: 'Pie chart with a center hole', example: 'Donut chart showing expense categories' },
                    { name: 'Sunburst', icon: 'fa-sun', description: 'Hierarchical data in concentric circles', example: 'Sunburst chart of sales by region and product' },
                    { name: 'Treemap', icon: 'fa-th-large', description: 'Nested rectangles for hierarchical data', example: 'Treemap of budget allocation by department' }
                ]
            },
            {
                category: 'Trends & Time Series',
                charts: [
                    { name: 'Area Chart', icon: 'fa-area-chart', description: 'Show cumulative totals over time', example: 'Area chart of cumulative sales' },
                    { name: 'Line Chart', icon: 'fa-chart-line', description: 'Track changes over continuous intervals', example: 'Monthly temperature trends' },
                    { name: 'Candlestick', icon: 'fa-chart-candlestick', description: 'Financial data (open, high, low, close)', example: 'Stock price movements over time' }
                ]
            },
            {
                category: 'Advanced & Specialized',
                charts: [
                    { name: '3D Scatter', icon: 'fa-cube', description: 'Three-dimensional scatter plot', example: 'Show 3D relationship between height, weight, and age' },
                    { name: 'Funnel Chart', icon: 'fa-filter', description: 'Visualize progressive reduction in stages', example: 'Sales funnel from leads to conversions' },
                    { name: 'Waterfall', icon: 'fa-water', description: 'Show cumulative effect of sequential values', example: 'Profit breakdown from revenue to net income' },
                    { name: 'Polar Chart', icon: 'fa-circle-dot', description: 'Circular coordinate system visualization', example: 'Wind direction and speed distribution' }
                ]
            },
            {
                category: 'Statistical & Analysis',
                charts: [
                    { name: 'Density Plot', icon: 'fa-wave-square', description: 'Smooth distribution estimate', example: 'Density plot of test scores' },
                    { name: 'Strip Plot', icon: 'fa-grip-lines', description: 'Show individual data points by category', example: 'Strip plot of scores by class' },
                    { name: 'Parallel Coordinates', icon: 'fa-stream', description: 'Multivariate data visualization', example: 'Compare multiple features across samples' },
                    { name: 'Contour Plot', icon: 'fa-mountain', description: 'Show 3D surface on 2D plane', example: 'Elevation contours or probability density' }
                ]
            }
        ];

        let html = '<div class="chart-gallery-content">';
        html += '<h3><i class="fas fa-palette"></i> Available Visualization Types</h3>';
        html += '<p class="gallery-intro">Choose from 25+ chart types to analyze and visualize your data. Click any example to try it!</p>';
        
        chartTypes.forEach(category => {
            html += `<div class="chart-category">`;
            html += `<h4><i class="fas fa-folder-open"></i> ${category.category}</h4>`;
            html += `<div class="charts-grid">`;
            
            category.charts.forEach(chart => {
                html += `
                    <div class="chart-card">
                        <div class="chart-icon">
                            <i class="fas ${chart.icon}"></i>
                        </div>
                        <div class="chart-info">
                            <h5>${chart.name}</h5>
                            <p class="chart-description">${chart.description}</p>
                            <div class="chart-example">
                                <button class="try-chart-btn" onclick="window.pyDataAssistant.tryChartExample('${chart.example.replace(/'/g, "\\'")}')">
                                    <i class="fas fa-play"></i> Try: "${chart.example}"
                                </button>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += `</div></div>`;
        });
        
        html += '<div class="chart-tips">';
        html += '<h4><i class="fas fa-lightbulb"></i> Pro Tips</h4>';
        html += '<ul>';
        html += '<li><strong>Be specific:</strong> Include column names in your query for better results</li>';
        html += '<li><strong>Multiple dimensions:</strong> Use "color=column" or "size=column" for richer visualizations</li>';
        html += '<li><strong>Time series:</strong> Mention time-based columns for automatic time series handling</li>';
        html += '<li><strong>Custom queries:</strong> Ask naturally - "Show me X grouped by Y with Z as colors"</li>';
        html += '</ul>';
        html += '</div>';
        
        html += '</div>';
        return html;
    }

    tryChartExample(exampleQuery) {
        if (!this.queryInput) return;
        this.queryInput.value = exampleQuery;
        this.updateCharCount();
        // Scroll to chat input
        this.queryInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        this.queryInput.focus();
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
        console.log('Data keys:', Object.keys(data || {}));
        console.log('Message:', message);
        
        // Handle different response types
        switch (response_type) {
            case 'plot':
                console.log('Handling plot response');
                console.log('data.data:', data.data);
                console.log('data.type:', data.type);
                
                // Check if data has MIME type wrapper (e.g., {'application/vnd.plotly.v1+json': {...}})
                let plotData = data;
                if (data['application/vnd.plotly.v1+json']) {
                    console.log('Detected Plotly MIME type wrapper, extracting...');
                    plotData = { data: data['application/vnd.plotly.v1+json'], title: data.title, type: data.type };
                } else if (data['text/html']) {
                    console.log('Detected HTML wrapper, extracting...');
                    // Skip HTML and use the plotly json if available
                    if (data['application/vnd.plotly.v1+json']) {
                        plotData = { data: data['application/vnd.plotly.v1+json'], title: data.title, type: data.type };
                    }
                }
                
                this.addPlotMessage(plotData, message);
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
        
        // Format the message (insights) nicely
        const formattedMessage = message ? this.formatAnalysisText(message) : '';
        
        messageDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-chart-bar"></i></div>
            <div class="message-content">
                <div class="plot-container">
                    <h4>${title}</h4>
                    <div id="${plotId}" class="plot-area"></div>
                    ${formattedMessage ? `<div class="analysis-insights">${formattedMessage}</div>` : ''}
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

    // Settings Management
    openSettings() {
        const modal = document.getElementById('settingsModal');
        if (modal) {
            this.loadSettings();
            modal.style.display = 'flex';
        }
    }

    closeSettings() {
        const modal = document.getElementById('settingsModal');
        if (modal) modal.style.display = 'none';
    }

    loadSettings() {
        // Load saved settings from localStorage
        const settings = JSON.parse(localStorage.getItem('pydataSettings') || '{}');
        
        if (settings.theme) document.getElementById('themeSelect').value = settings.theme;
        if (settings.fontSize) document.getElementById('fontSizeSelect').value = settings.fontSize;
        if (settings.chartTheme) document.getElementById('chartThemeSelect').value = settings.chartTheme;
        if (settings.maxRowsDisplay) document.getElementById('maxRowsDisplay').value = settings.maxRowsDisplay;
        if (settings.responseStyle) document.getElementById('responseStyle').value = settings.responseStyle;
        
        if (settings.autoRenderCharts !== undefined) document.getElementById('autoRenderCharts').checked = settings.autoRenderCharts;
        if (settings.showDataLabels !== undefined) document.getElementById('showDataLabels').checked = settings.showDataLabels;
        if (settings.cacheResults !== undefined) document.getElementById('cacheResults').checked = settings.cacheResults;
        if (settings.showCodeBlocks !== undefined) document.getElementById('showCodeBlocks').checked = settings.showCodeBlocks;
        if (settings.autoSuggestions !== undefined) document.getElementById('autoSuggestions').checked = settings.autoSuggestions;
    }

    saveSettings() {
        const settings = {
            theme: document.getElementById('themeSelect').value,
            fontSize: document.getElementById('fontSizeSelect').value,
            chartTheme: document.getElementById('chartThemeSelect').value,
            maxRowsDisplay: document.getElementById('maxRowsDisplay').value,
            responseStyle: document.getElementById('responseStyle').value,
            autoRenderCharts: document.getElementById('autoRenderCharts').checked,
            showDataLabels: document.getElementById('showDataLabels').checked,
            cacheResults: document.getElementById('cacheResults').checked,
            showCodeBlocks: document.getElementById('showCodeBlocks').checked,
            autoSuggestions: document.getElementById('autoSuggestions').checked,
            anonymizeData: document.getElementById('anonymizeData').checked
        };
        
        localStorage.setItem('pydataSettings', JSON.stringify(settings));
        
        // Apply theme immediately
        this.applyTheme(settings.theme);
        
        // Apply font size
        document.documentElement.style.fontSize = settings.fontSize === 'small' ? '14px' : 
                                                    settings.fontSize === 'large' ? '18px' : '16px';
        
        this.closeSettings();
        this.showNotification('Settings saved successfully!', 'success');
    }

    resetSettings() {
        if (confirm('Are you sure you want to reset all settings to default?')) {
            localStorage.removeItem('pydataSettings');
            this.loadSettings();
            this.showNotification('Settings reset to default', 'info');
        }
    }

    applyTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.add('dark-theme');
        } else if (theme === 'light') {
            document.body.classList.remove('dark-theme');
        } else if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.body.classList.toggle('dark-theme', prefersDark);
        }
    }

    clearAllData() {
        if (confirm('⚠️ This will delete all session data and cannot be undone. Continue?')) {
            localStorage.clear();
            sessionStorage.clear();
            window.location.reload();
        }
    }

    openHelp() {
        const modal = document.getElementById('helpModal');
        const helpContent = document.getElementById('helpContent');
        
        if (modal && helpContent) {
            helpContent.innerHTML = this.generateHelpContent();
            modal.style.display = 'flex';
        }
    }

    closeHelp() {
        const modal = document.getElementById('helpModal');
        if (modal) modal.style.display = 'none';
    }

    generateHelpContent() {
        return `
            <div class="help-sections">
                <section class="help-section">
                    <h3><i class="fas fa-rocket"></i> Getting Started</h3>
                    <ol>
                        <li>Upload your CSV file by dragging and dropping or clicking the upload button</li>
                        <li>Wait for the data preview to load</li>
                        <li>Click on any suggested analysis or type your own question</li>
                        <li>View results, charts, and insights in the chat interface</li>
                    </ol>
                </section>

                <section class="help-section">
                    <h3><i class="fas fa-chart-line"></i> Basic Analysis Examples</h3>
                    <div class="example-queries">
                        <div class="example-query">
                            <code>Show me the distribution of [column_name]</code>
                            <p>Creates a histogram showing data distribution</p>
                        </div>
                        <div class="example-query">
                            <code>Create a scatter plot of [column1] vs [column2]</code>
                            <p>Visualizes relationship between two variables</p>
                        </div>
                        <div class="example-query">
                            <code>Show correlation heatmap</code>
                            <p>Displays correlations between numeric columns</p>
                        </div>
                        <div class="example-query">
                            <code>Analyze missing values</code>
                            <p>Identifies and visualizes missing data patterns</p>
                        </div>
                    </div>
                </section>

                <section class="help-section">
                    <h3><i class="fas fa-flask"></i> Advanced Statistical Analysis</h3>
                    <p style="margin-bottom: 1rem; color: var(--primary-color); font-weight: 600;">
                        🎉 NEW FEATURES! Professional statistical analysis suite with 11 powerful tools
                    </p>
                    
                    <div class="stats-categories">
                        <div class="stats-category">
                            <h4><i class="fas fa-vial"></i> Hypothesis Testing</h4>
                            <div class="example-queries">
                                <div class="example-query">
                                    <code>Test if [column] is normally distributed</code>
                                    <p>Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling, KS tests</p>
                                </div>
                                <div class="example-query">
                                    <code>Perform t-test between [column1] and [column2]</code>
                                    <p>Independent/paired t-tests with Cohen's d effect size</p>
                                </div>
                                <div class="example-query">
                                    <code>Compare [column] across different [category] groups using ANOVA</code>
                                    <p>One-way ANOVA with eta-squared effect size</p>
                                </div>
                                <div class="example-query">
                                    <code>Chi-square test between [category1] and [category2]</code>
                                    <p>Independence test with Cramér's V</p>
                                </div>
                                <div class="example-query">
                                    <code>Test correlation between [column1] and [column2]</code>
                                    <p>Pearson, Spearman, and Kendall correlation tests</p>
                                </div>
                            </div>
                        </div>

                        <div class="stats-category">
                            <h4><i class="fas fa-chart-line-up"></i> Regression Analysis</h4>
                            <div class="example-queries">
                                <div class="example-query">
                                    <code>Perform linear regression with [column1] predicting [column2]</code>
                                    <p>Includes R², RMSE, coefficients, and residual analysis</p>
                                </div>
                                <div class="example-query">
                                    <code>Fit polynomial regression (degree 3) for [x] and [y]</code>
                                    <p>Models non-linear relationships with polynomial curves</p>
                                </div>
                                <div class="example-query">
                                    <code>Logistic regression for binary classification of [target]</code>
                                    <p>Binary outcome prediction with accuracy metrics</p>
                                </div>
                            </div>
                        </div>

                        <div class="stats-category">
                            <h4><i class="fas fa-bullseye"></i> Outlier Detection</h4>
                            <div class="example-queries">
                                <div class="example-query">
                                    <code>Detect outliers in [column] using all methods</code>
                                    <p>IQR, Z-score, Modified Z-score (MAD), Isolation Forest</p>
                                </div>
                                <div class="example-query">
                                    <code>Find outliers using IQR method in [column]</code>
                                    <p>Interquartile range based outlier detection</p>
                                </div>
                                <div class="example-query">
                                    <code>Use Isolation Forest to detect anomalies in [column]</code>
                                    <p>Machine learning based outlier detection</p>
                                </div>
                            </div>
                        </div>

                        <div class="stats-category">
                            <h4><i class="fas fa-wave-square"></i> Distribution Fitting</h4>
                            <div class="example-queries">
                                <div class="example-query">
                                    <code>Fit distributions to [column] and find best fit</code>
                                    <p>Tests: normal, exponential, gamma, beta, lognormal, weibull, uniform, chi2, t, f, pareto, logistic, gumbel</p>
                                </div>
                                <div class="example-query">
                                    <code>Compare normal and exponential distributions for [column]</code>
                                    <p>Includes AIC, BIC, log-likelihood, and KS test</p>
                                </div>
                            </div>
                        </div>

                        <div class="stats-category">
                            <h4><i class="fas fa-clock"></i> Time Series Analysis</h4>
                            <div class="example-queries">
                                <div class="example-query">
                                    <code>Test stationarity of [time_series_column]</code>
                                    <p>Augmented Dickey-Fuller test for time series data</p>
                                </div>
                                <div class="example-query">
                                    <code>Test if [column1] Granger-causes [column2]</code>
                                    <p>Predictive causality analysis for time series</p>
                                </div>
                            </div>
                        </div>

                        <div class="stats-category">
                            <h4><i class="fas fa-calculator"></i> Summary Statistics</h4>
                            <div class="example-queries">
                                <div class="example-query">
                                    <code>Get comprehensive statistics for [column]</code>
                                    <p>Mean, median, mode, std, variance, skewness, kurtosis, quartiles, range</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                <section class="help-section">
                    <h3><i class="fas fa-chart-pie"></i> Visualization Types</h3>
                    <div class="viz-grid">
                        <div class="viz-type">
                            <strong>Histogram</strong>: Distribution analysis
                        </div>
                        <div class="viz-type">
                            <strong>Box Plot</strong>: Outlier detection
                        </div>
                        <div class="viz-type">
                            <strong>Scatter Plot</strong>: Relationships
                        </div>
                        <div class="viz-type">
                            <strong>Line Chart</strong>: Trends over time
                        </div>
                        <div class="viz-type">
                            <strong>Bar Chart</strong>: Categorical comparison
                        </div>
                        <div class="viz-type">
                            <strong>Pie Chart</strong>: Proportions
                        </div>
                        <div class="viz-type">
                            <strong>Heatmap</strong>: Correlations
                        </div>
                        <div class="viz-type">
                            <strong>Violin Plot</strong>: Distribution comparison
                        </div>
                        <div class="viz-type">
                            <strong>Bubble Chart</strong>: 3D relationships
                        </div>
                        <div class="viz-type">
                            <strong>Sunburst</strong>: Hierarchical data
                        </div>
                    </div>
                </section>

                <section class="help-section">
                    <h3><i class="fas fa-lightbulb"></i> Tips & Best Practices</h3>
                    <ul>
                        <li><strong>Be specific</strong>: Mention exact column names in your queries</li>
                        <li><strong>Use suggestions</strong>: Click suggested analyses after upload</li>
                        <li><strong>Check data types</strong>: Ensure numeric columns for statistical tests</li>
                        <li><strong>Handle missing data</strong>: Address missing values before analysis</li>
                        <li><strong>Verify assumptions</strong>: Test normality before parametric tests</li>
                        <li><strong>Interpret effect sizes</strong>: Don't rely solely on p-values</li>
                        <li><strong>Export results</strong>: Save your analysis using the export button</li>
                    </ul>
                </section>

                <section class="help-section">
                    <h3><i class="fas fa-info-circle"></i> API Information</h3>
                    <p>For developers: Full API documentation is available at 
                    <a href="/docs" target="_blank" style="color: var(--primary-color);">/docs</a></p>
                    <p>Statistical Analysis endpoints: <code>/api/v1/statistical-analysis/*</code></p>
                </section>
            </div>
            
            <style>
                .help-sections { padding: 1rem; }
                .help-section { margin-bottom: 2rem; }
                .help-section h3 { color: var(--primary-color); margin-bottom: 1rem; }
                .help-section h4 { color: var(--text-primary); margin: 1rem 0 0.5rem 0; font-size: 1.1em; }
                .help-section ol, .help-section ul { padding-left: 1.5rem; }
                .help-section li { margin: 0.5rem 0; line-height: 1.6; }
                .example-queries { display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.5rem; }
                .example-query { 
                    background: var(--surface-color); 
                    padding: 0.75rem; 
                    border-radius: 8px;
                    border-left: 3px solid var(--primary-color);
                }
                .example-query code { 
                    display: block;
                    background: var(--background-color); 
                    padding: 0.5rem; 
                    border-radius: 4px;
                    margin-bottom: 0.5rem;
                    font-family: 'Courier New', monospace;
                    color: var(--primary-color);
                }
                .example-query p { 
                    margin: 0; 
                    color: var(--text-secondary);
                    font-size: 0.9em;
                }
                .stats-categories { display: flex; flex-direction: column; gap: 1.5rem; }
                .stats-category {
                    background: var(--surface-color);
                    padding: 1rem;
                    border-radius: 8px;
                    border: 1px solid var(--border-color);
                }
                .viz-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 0.75rem;
                    margin-top: 0.5rem;
                }
                .viz-type {
                    background: var(--surface-color);
                    padding: 0.75rem;
                    border-radius: 6px;
                    border-left: 3px solid var(--secondary-color);
                }
            </style>
        `;
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Export Data
    async exportData() {
        if (!this.currentSessionId || !this.currentDataPreview) {
            this.showNotification('No data available to export', 'error');
            return;
        }

        try {
            this.showLoading('Preparing export...');
            
            // Fetch full dataset
            const response = await fetch(`${this.apiBaseUrl}/session/${this.currentSessionId}/data?page=1&page_size=999999`);
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.message || 'Failed to fetch data');
            }

            // Convert to CSV
            const csv = this.convertToCSV(result.data, result.columns);
            
            // Download
            this.downloadFile(csv, `pydata-export-${new Date().toISOString().split('T')[0]}.csv`, 'text/csv');
            
            this.hideLoading();
            this.showNotification('Data exported successfully!', 'success');
        } catch (error) {
            this.hideLoading();
            this.showNotification(`Export failed: ${error.message}`, 'error');
            console.error('Export error:', error);
        }
    }

    convertToCSV(data, columns) {
        if (!data || data.length === 0) return '';
        
        // Header row
        let csv = columns.map(col => `"${col}"`).join(',') + '\n';
        
        // Data rows
        data.forEach(row => {
            const values = columns.map(col => {
                let value = row[col];
                if (value === null || value === undefined) value = '';
                // Escape quotes and wrap in quotes if needed
                value = String(value).replace(/"/g, '""');
                if (value.includes(',') || value.includes('"') || value.includes('\n')) {
                    value = `"${value}"`;
                }
                return value;
            });
            csv += values.join(',') + '\n';
        });
        
        return csv;
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing PyData Assistant...');
    window.pyDataAssistant = new PyDataAssistant();
});