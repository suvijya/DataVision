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
<<<<<<< Updated upstream
=======
            case 'chartgallery':
                content = this.renderChartGallery();
                break;
            case 'analytics':
                content = this.renderAnalyticsPanel();
                break;
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
=======
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

    renderAnalyticsPanel() {
        const preview = this.currentDataPreview;
        if (!preview) return '<p>No data loaded</p>';
        
        const numericCols = preview.columns.filter(col => 
            preview.dtypes[col] && (preview.dtypes[col].includes('int') || preview.dtypes[col].includes('float'))
        );
        const categoricalCols = preview.columns.filter(col => 
            preview.dtypes[col] && preview.dtypes[col].includes('object')
        );
        const allCols = preview.columns || [];
        
        let html = `
            <div class="analytics-panel">
                <div class="analytics-header">
                    <h3><i class="fas fa-brain"></i> Advanced Analytics Engine</h3>
                    <p class="analytics-intro">Perform machine learning, statistical tests, and advanced data analysis</p>
                </div>
                
                <!-- Analytics Categories -->
                <div class="analytics-categories">
                    <!-- Predictive Analytics -->
                    <div class="analytics-card">
                        <div class="card-header">
                            <h4><i class="fas fa-chart-line"></i> Predictive Analytics</h4>
                            <span class="badge">ML</span>
                        </div>
                        <div class="card-content">
                            <div class="analytics-method">
                                <h5>Linear Regression</h5>
                                <p>Predict continuous values based on features</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Target Column:</label>
                                        <select id="linearRegressionTarget" class="form-select">
                                            <option value="">Select target...</option>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Feature Columns:</label>
                                        <select id="linearRegressionFeatures" class="form-select" multiple>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                        <small>Hold Ctrl/Cmd to select multiple</small>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runLinearRegression()">
                                        <i class="fas fa-play"></i> Run Linear Regression
                                    </button>
                                </div>
                            </div>
                            
                            <div class="analytics-method">
                                <h5>Logistic Regression</h5>
                                <p>Binary or multiclass classification</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Target Column:</label>
                                        <select id="logisticRegressionTarget" class="form-select">
                                            <option value="">Select target...</option>
                                            ${allCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Feature Columns:</label>
                                        <select id="logisticRegressionFeatures" class="form-select" multiple>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runLogisticRegression()">
                                        <i class="fas fa-play"></i> Run Logistic Regression
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Clustering -->
                    <div class="analytics-card">
                        <div class="card-header">
                            <h4><i class="fas fa-project-diagram"></i> Clustering Analysis</h4>
                            <span class="badge">Unsupervised</span>
                        </div>
                        <div class="card-content">
                            <div class="analytics-method">
                                <h5>K-Means Clustering</h5>
                                <p>Partition data into k clusters</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Feature Columns:</label>
                                        <select id="kmeansFeatures" class="form-select" multiple>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Number of Clusters:</label>
                                        <input type="number" id="kmeansNClusters" class="form-input" value="3" min="2" max="10">
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runKMeans()">
                                        <i class="fas fa-play"></i> Run K-Means
                                    </button>
                                </div>
                            </div>
                            
                            <div class="analytics-method">
                                <h5>DBSCAN Clustering</h5>
                                <p>Density-based clustering (finds arbitrary shapes)</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Feature Columns:</label>
                                        <select id="dbscanFeatures" class="form-select" multiple>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-row">
                                        <div class="form-group">
                                            <label>Epsilon:</label>
                                            <input type="number" id="dbscanEps" class="form-input" value="0.5" step="0.1" min="0.1">
                                        </div>
                                        <div class="form-group">
                                            <label>Min Samples:</label>
                                            <input type="number" id="dbscanMinSamples" class="form-input" value="5" min="2">
                                        </div>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runDBSCAN()">
                                        <i class="fas fa-play"></i> Run DBSCAN
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Dimensionality Reduction -->
                    <div class="analytics-card">
                        <div class="card-header">
                            <h4><i class="fas fa-compress-arrows-alt"></i> Dimensionality Reduction</h4>
                            <span class="badge">Visualization</span>
                        </div>
                        <div class="card-content">
                            <div class="analytics-method">
                                <h5>PCA (Principal Component Analysis)</h5>
                                <p>Linear dimensionality reduction</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Feature Columns:</label>
                                        <select id="pcaFeatures" class="form-select" multiple>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Components:</label>
                                        <select id="pcaComponents" class="form-select">
                                            <option value="2" selected>2D</option>
                                            <option value="3">3D</option>
                                        </select>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runPCA()">
                                        <i class="fas fa-play"></i> Run PCA
                                    </button>
                                </div>
                            </div>
                            
                            <div class="analytics-method">
                                <h5>t-SNE</h5>
                                <p>Non-linear dimensionality reduction for visualization</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Feature Columns:</label>
                                        <select id="tsneFeatures" class="form-select" multiple>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-row">
                                        <div class="form-group">
                                            <label>Components:</label>
                                            <select id="tsneComponents" class="form-select">
                                                <option value="2" selected>2D</option>
                                                <option value="3">3D</option>
                                            </select>
                                        </div>
                                        <div class="form-group">
                                            <label>Perplexity:</label>
                                            <input type="number" id="tsnePerplexity" class="form-input" value="30" min="5" max="50">
                                        </div>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runTSNE()">
                                        <i class="fas fa-play"></i> Run t-SNE
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Outlier Detection -->
                    <div class="analytics-card">
                        <div class="card-header">
                            <h4><i class="fas fa-exclamation-triangle"></i> Outlier Detection</h4>
                            <span class="badge">Anomaly</span>
                        </div>
                        <div class="card-content">
                            <div class="analytics-method">
                                <h5>Isolation Forest</h5>
                                <p>Detect outliers using ensemble method</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Feature Columns:</label>
                                        <select id="outlierFeatures" class="form-select" multiple>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Contamination (expected % of outliers):</label>
                                        <input type="number" id="outlierContamination" class="form-input" value="0.1" step="0.01" min="0.01" max="0.5">
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runOutlierDetection()">
                                        <i class="fas fa-play"></i> Detect Outliers
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Statistical Tests -->
                    <div class="analytics-card">
                        <div class="card-header">
                            <h4><i class="fas fa-calculator"></i> Statistical Tests</h4>
                            <span class="badge">Inference</span>
                        </div>
                        <div class="card-content">
                            <div class="analytics-method">
                                <h5>T-Test</h5>
                                <p>Compare means of two groups</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Group Column:</label>
                                        <select id="ttestGroupCol" class="form-select">
                                            <option value="">Select column...</option>
                                            ${categoricalCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Value Column:</label>
                                        <select id="ttestValueCol" class="form-select">
                                            <option value="">Select column...</option>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runTTest()">
                                        <i class="fas fa-play"></i> Run T-Test
                                    </button>
                                </div>
                            </div>
                            
                            <div class="analytics-method">
                                <h5>ANOVA</h5>
                                <p>Compare means of 3+ groups</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Group Column:</label>
                                        <select id="anovaGroupCol" class="form-select">
                                            <option value="">Select column...</option>
                                            ${categoricalCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Value Column:</label>
                                        <select id="anovaValueCol" class="form-select">
                                            <option value="">Select column...</option>
                                            ${numericCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runANOVA()">
                                        <i class="fas fa-play"></i> Run ANOVA
                                    </button>
                                </div>
                            </div>
                            
                            <div class="analytics-method">
                                <h5>Chi-Square Test</h5>
                                <p>Test independence of categorical variables</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Column 1:</label>
                                        <select id="chiSquareCol1" class="form-select">
                                            <option value="">Select column...</option>
                                            ${categoricalCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <div class="form-group">
                                        <label>Column 2:</label>
                                        <select id="chiSquareCol2" class="form-select">
                                            <option value="">Select column...</option>
                                            ${categoricalCols.map(col => `<option value="${col}">${col}</option>`).join('')}
                                        </select>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runChiSquare()">
                                        <i class="fas fa-play"></i> Run Chi-Square
                                    </button>
                                </div>
                            </div>
                            
                            <div class="analytics-method">
                                <h5>Correlation Analysis</h5>
                                <p>Analyze relationships between numeric variables</p>
                                <div class="method-controls">
                                    <div class="form-group">
                                        <label>Method:</label>
                                        <select id="correlationMethod" class="form-select">
                                            <option value="pearson" selected>Pearson</option>
                                            <option value="spearman">Spearman</option>
                                            <option value="kendall">Kendall</option>
                                        </select>
                                    </div>
                                    <button class="analytics-btn primary" onclick="window.pyDataAssistant.runCorrelation()">
                                        <i class="fas fa-play"></i> Run Correlation
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        return html;
    }

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
=======

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

    
    // ==================== ADVANCED ANALYTICS METHODS ====================
    
    async runLinearRegression() {
        const target = document.getElementById('linearRegressionTarget').value;
        const featuresSelect = document.getElementById('linearRegressionFeatures');
        const features = Array.from(featuresSelect.selectedOptions).map(opt => opt.value);
        
        if (!target || features.length === 0) {
            this.showNotification('Please select target and at least one feature column', 'error');
            return;
        }
        
        await this.runAnalytics('regression', {
            session_id: this.currentSessionId,
            target_column: target,
            feature_columns: features,
            model_type: 'linear'
        });
    }
    
    async runLogisticRegression() {
        const target = document.getElementById('logisticRegressionTarget').value;
        const featuresSelect = document.getElementById('logisticRegressionFeatures');
        const features = Array.from(featuresSelect.selectedOptions).map(opt => opt.value);
        
        if (!target || features.length === 0) {
            this.showNotification('Please select target and at least one feature column', 'error');
            return;
        }
        
        await this.runAnalytics('regression', {
            session_id: this.currentSessionId,
            target_column: target,
            feature_columns: features,
            model_type: 'logistic'
        });
    }
    
    async runKMeans() {
        const featuresSelect = document.getElementById('kmeansFeatures');
        const features = Array.from(featuresSelect.selectedOptions).map(opt => opt.value);
        const nClusters = parseInt(document.getElementById('kmeansNClusters').value);
        
        if (features.length === 0) {
            this.showNotification('Please select at least one feature column', 'error');
            return;
        }
        
        await this.runAnalytics('clustering', {
            session_id: this.currentSessionId,
            feature_columns: features,
            algorithm: 'kmeans',
            n_clusters: nClusters
        });
    }
    
    async runDBSCAN() {
        const featuresSelect = document.getElementById('dbscanFeatures');
        const features = Array.from(featuresSelect.selectedOptions).map(opt => opt.value);
        const eps = parseFloat(document.getElementById('dbscanEps').value);
        const minSamples = parseInt(document.getElementById('dbscanMinSamples').value);
        
        if (features.length === 0) {
            this.showNotification('Please select at least one feature column', 'error');
            return;
        }
        
        await this.runAnalytics('clustering', {
            session_id: this.currentSessionId,
            feature_columns: features,
            algorithm: 'dbscan',
            eps: eps,
            min_samples: minSamples
        });
    }
    
    async runPCA() {
        const featuresSelect = document.getElementById('pcaFeatures');
        const features = Array.from(featuresSelect.selectedOptions).map(opt => opt.value);
        const nComponents = parseInt(document.getElementById('pcaComponents').value);
        
        if (features.length < 2) {
            this.showNotification('Please select at least 2 feature columns', 'error');
            return;
        }
        
        await this.runAnalytics('dimensionality-reduction', {
            session_id: this.currentSessionId,
            feature_columns: features,
            algorithm: 'pca',
            n_components: nComponents
        });
    }
    
    async runTSNE() {
        const featuresSelect = document.getElementById('tsneFeatures');
        const features = Array.from(featuresSelect.selectedOptions).map(opt => opt.value);
        const nComponents = parseInt(document.getElementById('tsneComponents').value);
        const perplexity = parseInt(document.getElementById('tsnePerplexity').value);
        
        if (features.length < 2) {
            this.showNotification('Please select at least 2 feature columns', 'error');
            return;
        }
        
        await this.runAnalytics('dimensionality-reduction', {
            session_id: this.currentSessionId,
            feature_columns: features,
            algorithm: 'tsne',
            n_components: nComponents,
            perplexity: perplexity
        });
    }
    
    async runOutlierDetection() {
        const featuresSelect = document.getElementById('outlierFeatures');
        const features = Array.from(featuresSelect.selectedOptions).map(opt => opt.value);
        const contamination = parseFloat(document.getElementById('outlierContamination').value);
        
        if (features.length === 0) {
            this.showNotification('Please select at least one feature column', 'error');
            return;
        }
        
        await this.runAnalytics('outliers', {
            session_id: this.currentSessionId,
            feature_columns: features,
            contamination: contamination
        });
    }
    
    async runTTest() {
        const groupCol = document.getElementById('ttestGroupCol').value;
        const valueCol = document.getElementById('ttestValueCol').value;
        
        if (!groupCol || !valueCol) {
            this.showNotification('Please select both group and value columns', 'error');
            return;
        }
        
        await this.runAnalytics('statistical-test', {
            session_id: this.currentSessionId,
            test_type: 'ttest',
            group_column: groupCol,
            value_column: valueCol
        });
    }
    
    async runANOVA() {
        const groupCol = document.getElementById('anovaGroupCol').value;
        const valueCol = document.getElementById('anovaValueCol').value;
        
        if (!groupCol || !valueCol) {
            this.showNotification('Please select both group and value columns', 'error');
            return;
        }
        
        await this.runAnalytics('statistical-test', {
            session_id: this.currentSessionId,
            test_type: 'anova',
            group_column: groupCol,
            value_column: valueCol
        });
    }
    
    async runChiSquare() {
        const col1 = document.getElementById('chiSquareCol1').value;
        const col2 = document.getElementById('chiSquareCol2').value;
        
        if (!col1 || !col2) {
            this.showNotification('Please select both columns', 'error');
            return;
        }
        
        await this.runAnalytics('statistical-test', {
            session_id: this.currentSessionId,
            test_type: 'chi_square',
            column1: col1,
            column2: col2
        });
    }
    
    async runCorrelation() {
        const method = document.getElementById('correlationMethod').value;
        
        await this.runAnalytics('statistical-test', {
            session_id: this.currentSessionId,
            test_type: 'correlation',
            method: method
        });
    }
    
    async runAnalytics(endpoint, requestData) {
        this.showLoading(`Running ${endpoint} analysis...`);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/analytics/${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                let errorMessage = `HTTP ${response.status}`;
                try {
                    const errorData = JSON.parse(errorText);
                    errorMessage = errorData.detail || errorData.message || errorMessage;
                } catch (e) {
                    errorMessage = errorText || errorMessage;
                }
                throw new Error(errorMessage);
            }
            
            const result = await response.json();
            this.hideLoading();
            
            // Display results in chat
            this.displayAnalyticsResult(result);
            
            // Show success notification
            this.showNotification(result.message || 'Analysis completed successfully', 'success');
            
            // Scroll to chat to see results
            if (this.chatSection) {
                this.chatSection.scrollIntoView({ behavior: 'smooth' });
            }
            
        } catch (error) {
            this.hideLoading();
            console.error('Analytics error:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
            this.addMessage('assistant', `❌ Analysis failed: ${error.message}`);
        }
    }
    
    displayAnalyticsResult(result) {
        if (!result.success) {
            this.addMessage('assistant', `❌ ${result.message}`);
            return;
        }
        
        const data = result.data;
        
        // Format message with key metrics
        let message = `### 📊 ${data.algorithm || data.test || data.model_type || 'Analysis'} Results\n\n`;
        
        // Add metrics
        if (data.metrics) {
            message += '**Metrics:**\n';
            for (const [key, value] of Object.entries(data.metrics)) {
                if (typeof value === 'number') {
                    message += `• ${key.replace(/_/g, ' ')}: ${value.toFixed(4)}\n`;
                } else {
                    message += `• ${key.replace(/_/g, ' ')}: ${value}\n`;
                }
            }
            message += '\n';
        }
        
        // Add interpretation if available
        if (data.interpretation) {
            message += `**Interpretation:**\n${data.interpretation}\n\n`;
        }
        
        // Add summary info
        if (data.n_clusters !== undefined) {
            message += `**Found ${data.n_clusters} clusters**\n\n`;
        }
        
        if (data.n_outliers !== undefined) {
            message += `**Detected ${data.n_outliers} outliers (${data.outlier_percentage})**\n\n`;
        }
        
        if (data.explained_variance_ratio) {
            message += `**Explained Variance:**\n`;
            data.explained_variance_ratio.forEach((v, i) => {
                message += `• PC${i+1}: ${(v * 100).toFixed(2)}%\n`;
            });
            message += `\n**Total: ${data.total_variance_explained}**\n\n`;
        }
        
        // Add cluster sizes
        if (data.cluster_sizes) {
            message += '**Cluster Sizes:**\n';
            for (const [cluster, info] of Object.entries(data.cluster_sizes)) {
                message += `• ${cluster}: ${info.size} samples (${info.percentage})\n`;
            }
            message += '\n';
        }
        
        // Add group statistics
        if (data.group_statistics) {
            message += '**Group Statistics:**\n';
            for (const [group, stats] of Object.entries(data.group_statistics)) {
                message += `\n**${group}:**\n`;
                for (const [key, value] of Object.entries(stats)) {
                    if (typeof value === 'number') {
                        message += `  • ${key}: ${value.toFixed(4)}\n`;
                    } else {
                        message += `  • ${key}: ${value}\n`;
                    }
                }
            }
            message += '\n';
        }
        
        // Add coefficients for regression
        if (data.coefficients) {
            message += '**Feature Coefficients:**\n';
            for (const [feature, coef] of Object.entries(data.coefficients)) {
                message += `• ${feature}: ${coef.toFixed(4)}\n`;
            }
            message += '\n';
        }
        
        // Add top correlations
        if (data.top_correlations) {
            message += '**Top 5 Correlations:**\n';
            data.top_correlations.slice(0, 5).forEach((corr, i) => {
                message += `${i+1}. ${corr.var1} ↔ ${corr.var2}: ${corr.correlation.toFixed(4)}\n`;
            });
            message += '\n';
        }
        
        message += `\n_Full details available in visualization below_`;
        
        // Add formatted message
        this.addFormattedMessage('assistant', message);
        
        // Add visualization if available
        if (result.visualization) {
            this.addPlotMessage({ data: result.visualization }, '');
        }
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
>>>>>>> Stashed changes
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing PyData Assistant...');
    window.pyDataAssistant = new PyDataAssistant();
});