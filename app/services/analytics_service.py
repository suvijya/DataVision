"""
Advanced Analytics Service
Provides ML models, statistical tests, and advanced data analysis capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, davies_bouldin_score
)
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsResult:
    """Standardized analytics result."""
    analysis_type: str
    success: bool
    data: Dict[str, Any]
    visualization: Optional[Dict[str, Any]] = None
    message: str = ""
    error: Optional[str] = None


class AdvancedAnalyticsService:
    """Advanced analytics and machine learning service."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    # ==================== PREDICTIVE ANALYTICS ====================
    
    def linear_regression(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        test_size: float = 0.2
    ) -> AnalyticsResult:
        """
        Perform linear regression analysis.
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
            feature_cols: List of feature column names
            test_size: Test set proportion
            
        Returns:
            AnalyticsResult with model metrics and predictions
        """
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df[target_col].fillna(df[target_col].mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, model.coef_))
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color='blue', opacity=0.6)
            ))
            
            # Add perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'Linear Regression: Actual vs Predicted {target_col}',
                xaxis_title=f'Actual {target_col}',
                yaxis_title=f'Predicted {target_col}',
                hovermode='closest'
            )
            
            return AnalyticsResult(
                analysis_type='linear_regression',
                success=True,
                data={
                    'model_type': 'Linear Regression',
                    'target': target_col,
                    'features': feature_cols,
                    'metrics': {
                        'r2_score': float(r2),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'train_samples': len(X_train),
                        'test_samples': len(X_test)
                    },
                    'coefficients': {k: float(v) for k, v in feature_importance.items()},
                    'intercept': float(model.intercept_),
                    'predictions': {
                        'actual': y_test.tolist()[:100],  # Limit for JSON
                        'predicted': y_pred.tolist()[:100]
                    }
                },
                visualization=fig.to_dict(),
                message=f"Linear Regression completed. R² = {r2:.4f}, RMSE = {rmse:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Linear regression error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='linear_regression',
                success=False,
                data={},
                error=str(e),
                message=f"Linear regression failed: {str(e)}"
            )
    
    def logistic_regression(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        test_size: float = 0.2
    ) -> AnalyticsResult:
        """Perform logistic regression for binary classification."""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            y = df[target_col]
            
            # Encode target if necessary
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                classes = le.classes_
            else:
                classes = np.unique(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle binary vs multiclass
            avg_method = 'binary' if len(classes) == 2 else 'weighted'
            precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
            recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
            
            # Confusion matrix visualization
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f'Predicted {c}' for c in classes],
                y=[f'Actual {c}' for c in classes],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            
            fig.update_layout(
                title=f'Confusion Matrix - {target_col}',
                xaxis_title='Predicted Class',
                yaxis_title='Actual Class'
            )
            
            return AnalyticsResult(
                analysis_type='logistic_regression',
                success=True,
                data={
                    'model_type': 'Logistic Regression',
                    'target': target_col,
                    'features': feature_cols,
                    'classes': [str(c) for c in classes],
                    'metrics': {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'train_samples': len(X_train),
                        'test_samples': len(X_test)
                    },
                    'confusion_matrix': cm.tolist(),
                    'feature_importance': dict(zip(feature_cols, model.coef_[0].tolist()))
                },
                visualization=fig.to_dict(),
                message=f"Logistic Regression completed. Accuracy = {accuracy:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Logistic regression error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='logistic_regression',
                success=False,
                data={},
                error=str(e),
                message=f"Logistic regression failed: {str(e)}"
            )
    
    # ==================== CLUSTERING ANALYSIS ====================
    
    def kmeans_clustering(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_clusters: int = 3,
        visualize_2d: bool = True
    ) -> AnalyticsResult:
        """Perform K-means clustering analysis."""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add clusters to dataframe
            df_clustered = df.copy()
            df_clustered['Cluster'] = clusters
            
            # Calculate metrics
            silhouette = silhouette_score(X_scaled, clusters)
            davies_bouldin = davies_bouldin_score(X_scaled, clusters)
            inertia = kmeans.inertia_
            
            # Create visualization
            if len(feature_cols) >= 2 and visualize_2d:
                # Use first two features or PCA
                if len(feature_cols) > 2:
                    pca = PCA(n_components=2)
                    X_viz = pca.fit_transform(X_scaled)
                    x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})'
                    y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
                else:
                    X_viz = X_scaled
                    x_label = feature_cols[0]
                    y_label = feature_cols[1]
                
                fig = go.Figure()
                
                # Plot each cluster
                for i in range(n_clusters):
                    mask = clusters == i
                    fig.add_trace(go.Scatter(
                        x=X_viz[mask, 0],
                        y=X_viz[mask, 1],
                        mode='markers',
                        name=f'Cluster {i}',
                        marker=dict(size=8, opacity=0.6)
                    ))
                
                # Plot centroids
                if len(feature_cols) > 2:
                    centroids_viz = pca.transform(kmeans.cluster_centers_)
                else:
                    centroids_viz = kmeans.cluster_centers_
                
                fig.add_trace(go.Scatter(
                    x=centroids_viz[:, 0],
                    y=centroids_viz[:, 1],
                    mode='markers',
                    name='Centroids',
                    marker=dict(
                        size=15,
                        color='black',
                        symbol='x',
                        line=dict(width=2, color='white')
                    )
                ))
                
                fig.update_layout(
                    title=f'K-Means Clustering (k={n_clusters})',
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    hovermode='closest'
                )
            else:
                fig = None
            
            # Cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_data = df_clustered[df_clustered['Cluster'] == i]
                cluster_stats[f'Cluster_{i}'] = {
                    'size': len(cluster_data),
                    'percentage': f"{len(cluster_data) / len(df) * 100:.2f}%"
                }
            
            return AnalyticsResult(
                analysis_type='kmeans_clustering',
                success=True,
                data={
                    'algorithm': 'K-Means',
                    'n_clusters': n_clusters,
                    'features': feature_cols,
                    'metrics': {
                        'silhouette_score': float(silhouette),
                        'davies_bouldin_index': float(davies_bouldin),
                        'inertia': float(inertia)
                    },
                    'cluster_sizes': cluster_stats,
                    'cluster_labels': clusters.tolist()
                },
                visualization=fig.to_dict() if fig else None,
                message=f"K-Means clustering completed. Silhouette Score = {silhouette:.4f}"
            )
            
        except Exception as e:
            logger.error(f"K-means clustering error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='kmeans_clustering',
                success=False,
                data={},
                error=str(e),
                message=f"K-means clustering failed: {str(e)}"
            )
    
    def dbscan_clustering(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        eps: float = 0.5,
        min_samples: int = 5
    ) -> AnalyticsResult:
        """Perform DBSCAN clustering (density-based)."""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            
            # Calculate metrics
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            # Silhouette score (only if we have clusters)
            if n_clusters > 1:
                # Exclude noise points for silhouette calculation
                mask = clusters != -1
                if mask.sum() > 0:
                    silhouette = silhouette_score(X_scaled[mask], clusters[mask])
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            # Visualization
            if len(feature_cols) >= 2:
                if len(feature_cols) > 2:
                    pca = PCA(n_components=2)
                    X_viz = pca.fit_transform(X_scaled)
                    x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})'
                    y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
                else:
                    X_viz = X_scaled
                    x_label = feature_cols[0]
                    y_label = feature_cols[1]
                
                fig = px.scatter(
                    x=X_viz[:, 0],
                    y=X_viz[:, 1],
                    color=clusters.astype(str),
                    title=f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})',
                    labels={'x': x_label, 'y': y_label, 'color': 'Cluster'}
                )
                fig.update_traces(marker=dict(size=8, opacity=0.7))
            else:
                fig = None
            
            # Cluster statistics
            cluster_stats = {}
            unique_clusters = set(clusters)
            for cluster_id in unique_clusters:
                size = list(clusters).count(cluster_id)
                cluster_stats[f'Cluster_{cluster_id}'] = {
                    'size': size,
                    'percentage': f"{size / len(df) * 100:.2f}%",
                    'is_noise': cluster_id == -1
                }
            
            return AnalyticsResult(
                analysis_type='dbscan_clustering',
                success=True,
                data={
                    'algorithm': 'DBSCAN',
                    'parameters': {'eps': eps, 'min_samples': min_samples},
                    'features': feature_cols,
                    'metrics': {
                        'n_clusters': n_clusters,
                        'n_noise_points': n_noise,
                        'noise_percentage': f"{n_noise / len(df) * 100:.2f}%",
                        'silhouette_score': float(silhouette) if silhouette != -1 else None
                    },
                    'cluster_sizes': cluster_stats,
                    'cluster_labels': clusters.tolist()
                },
                visualization=fig.to_dict() if fig else None,
                message=f"DBSCAN completed. Found {n_clusters} clusters and {n_noise} noise points."
            )
            
        except Exception as e:
            logger.error(f"DBSCAN clustering error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='dbscan_clustering',
                success=False,
                data={},
                error=str(e),
                message=f"DBSCAN clustering failed: {str(e)}"
            )
    
    # ==================== DIMENSIONALITY REDUCTION ====================
    
    def pca_analysis(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_components: int = 2
    ) -> AnalyticsResult:
        """Perform Principal Component Analysis."""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Create DataFrame with PCA components
            pca_cols = [f'PC{i+1}' for i in range(n_components)]
            df_pca = pd.DataFrame(X_pca, columns=pca_cols)
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            # Create visualizations
            if n_components >= 2:
                # 2D scatter plot
                fig = px.scatter(
                    df_pca,
                    x='PC1',
                    y='PC2',
                    title=f'PCA: First 2 Components (Explained Variance: {cumulative_var[1]:.2%})',
                    labels={
                        'PC1': f'PC1 ({explained_var[0]:.2%})',
                        'PC2': f'PC2 ({explained_var[1]:.2%})'
                    }
                )
                fig.update_traces(marker=dict(size=6, opacity=0.7))
                
                # Scree plot (variance explained)
                scree_fig = go.Figure()
                scree_fig.add_trace(go.Bar(
                    x=[f'PC{i+1}' for i in range(len(explained_var))],
                    y=explained_var * 100,
                    name='Individual',
                    marker_color='lightblue'
                ))
                scree_fig.add_trace(go.Scatter(
                    x=[f'PC{i+1}' for i in range(len(cumulative_var))],
                    y=cumulative_var * 100,
                    name='Cumulative',
                    mode='lines+markers',
                    line=dict(color='red', width=2)
                ))
                scree_fig.update_layout(
                    title='PCA Scree Plot - Explained Variance',
                    xaxis_title='Principal Component',
                    yaxis_title='Variance Explained (%)',
                    barmode='group'
                )
            else:
                fig = None
                scree_fig = None
            
            # Component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=pca_cols,
                index=feature_cols
            )
            
            return AnalyticsResult(
                analysis_type='pca',
                success=True,
                data={
                    'algorithm': 'PCA',
                    'n_components': n_components,
                    'original_features': feature_cols,
                    'explained_variance_ratio': explained_var.tolist(),
                    'cumulative_variance': cumulative_var.tolist(),
                    'total_variance_explained': f"{cumulative_var[-1]:.2%}",
                    'loadings': loadings.to_dict(),
                    'principal_components': df_pca.head(100).to_dict('records')  # Limit for JSON
                },
                visualization=fig.to_dict() if fig else None,
                message=f"PCA completed. {n_components} components explain {cumulative_var[-1]:.2%} of variance."
            )
            
        except Exception as e:
            logger.error(f"PCA error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='pca',
                success=False,
                data={},
                error=str(e),
                message=f"PCA failed: {str(e)}"
            )
    
    def tsne_analysis(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_components: int = 2,
        perplexity: int = 30
    ) -> AnalyticsResult:
        """Perform t-SNE dimensionality reduction."""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Limit samples for performance (t-SNE is computationally expensive)
            max_samples = 5000
            if len(X) > max_samples:
                logger.warning(f"Sampling {max_samples} points for t-SNE (original: {len(X)})")
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_scaled = X_scaled[indices]
            else:
                indices = None
            
            # Perform t-SNE
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Create DataFrame
            tsne_cols = [f'tSNE{i+1}' for i in range(n_components)]
            df_tsne = pd.DataFrame(X_tsne, columns=tsne_cols)
            
            # Visualization
            if n_components >= 2:
                fig = px.scatter(
                    df_tsne,
                    x='tSNE1',
                    y='tSNE2',
                    title=f't-SNE Visualization (perplexity={perplexity})',
                    labels={'tSNE1': 't-SNE Dimension 1', 'tSNE2': 't-SNE Dimension 2'}
                )
                fig.update_traces(marker=dict(size=5, opacity=0.6))
            else:
                fig = None
            
            return AnalyticsResult(
                analysis_type='tsne',
                success=True,
                data={
                    'algorithm': 't-SNE',
                    'n_components': n_components,
                    'perplexity': perplexity,
                    'original_features': feature_cols,
                    'samples_used': len(X_scaled),
                    'sampled': indices is not None,
                    'tsne_components': df_tsne.head(100).to_dict('records')
                },
                visualization=fig.to_dict() if fig else None,
                message=f"t-SNE completed with perplexity={perplexity}."
            )
            
        except Exception as e:
            logger.error(f"t-SNE error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='tsne',
                success=False,
                data={},
                error=str(e),
                message=f"t-SNE failed: {str(e)}"
            )
    
    # ==================== OUTLIER DETECTION ====================
    
    def isolation_forest(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        contamination: float = 0.1
    ) -> AnalyticsResult:
        """Detect outliers using Isolation Forest."""
        try:
            # Prepare data
            X = df[feature_cols].fillna(df[feature_cols].mean())
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            predictions = iso_forest.fit_predict(X_scaled)
            scores = iso_forest.score_samples(X_scaled)
            
            # -1 for outliers, 1 for inliers
            outliers = predictions == -1
            n_outliers = outliers.sum()
            
            # Add results to dataframe
            df_result = df.copy()
            df_result['Is_Outlier'] = outliers
            df_result['Anomaly_Score'] = scores
            
            # Visualization
            if len(feature_cols) >= 2:
                if len(feature_cols) > 2:
                    pca = PCA(n_components=2)
                    X_viz = pca.fit_transform(X_scaled)
                    x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})'
                    y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
                else:
                    X_viz = X_scaled
                    x_label = feature_cols[0]
                    y_label = feature_cols[1]
                
                fig = go.Figure()
                
                # Inliers
                fig.add_trace(go.Scatter(
                    x=X_viz[~outliers, 0],
                    y=X_viz[~outliers, 1],
                    mode='markers',
                    name='Normal',
                    marker=dict(size=6, color='blue', opacity=0.5)
                ))
                
                # Outliers
                fig.add_trace(go.Scatter(
                    x=X_viz[outliers, 0],
                    y=X_viz[outliers, 1],
                    mode='markers',
                    name='Outliers',
                    marker=dict(size=10, color='red', symbol='x')
                ))
                
                fig.update_layout(
                    title=f'Isolation Forest: Outlier Detection ({n_outliers} outliers found)',
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    hovermode='closest'
                )
            else:
                fig = None
            
            # Get outlier samples
            outlier_indices = np.where(outliers)[0]
            outlier_samples = df_result.iloc[outlier_indices].head(20).to_dict('records')
            
            return AnalyticsResult(
                analysis_type='isolation_forest',
                success=True,
                data={
                    'algorithm': 'Isolation Forest',
                    'contamination': contamination,
                    'features': feature_cols,
                    'n_outliers': int(n_outliers),
                    'outlier_percentage': f"{n_outliers / len(df) * 100:.2f}%",
                    'n_normal': int((~outliers).sum()),
                    'outlier_indices': outlier_indices.tolist()[:100],  # Limit
                    'outlier_samples': outlier_samples,
                    'anomaly_scores': {
                        'min': float(scores.min()),
                        'max': float(scores.max()),
                        'mean': float(scores.mean())
                    }
                },
                visualization=fig.to_dict() if fig else None,
                message=f"Found {n_outliers} outliers ({n_outliers / len(df) * 100:.2f}%) using Isolation Forest."
            )
            
        except Exception as e:
            logger.error(f"Isolation Forest error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='isolation_forest',
                success=False,
                data={},
                error=str(e),
                message=f"Outlier detection failed: {str(e)}"
            )
    
    # ==================== STATISTICAL TESTS ====================
    
    def ttest(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        test_type: str = 'independent'
    ) -> AnalyticsResult:
        """Perform T-test (independent or paired)."""
        try:
            # Get unique groups
            groups = df[group_col].unique()
            
            if len(groups) != 2:
                return AnalyticsResult(
                    analysis_type='ttest',
                    success=False,
                    data={},
                    error=f"T-test requires exactly 2 groups, found {len(groups)}",
                    message="T-test requires exactly 2 groups"
                )
            
            # Get data for each group
            group1_data = df[df[group_col] == groups[0]][value_col].dropna()
            group2_data = df[df[group_col] == groups[1]][value_col].dropna()
            
            # Perform t-test
            if test_type == 'independent':
                statistic, pvalue = stats.ttest_ind(group1_data, group2_data)
                test_name = "Independent Samples T-Test"
            else:
                statistic, pvalue = stats.ttest_rel(group1_data, group2_data)
                test_name = "Paired Samples T-Test"
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((group1_data.std()**2 + group2_data.std()**2) / 2)
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            # Visualization - Box plot comparison
            fig = go.Figure()
            fig.add_trace(go.Box(y=group1_data, name=str(groups[0])))
            fig.add_trace(go.Box(y=group2_data, name=str(groups[1])))
            fig.update_layout(
                title=f'{test_name}: {value_col} by {group_col}',
                yaxis_title=value_col,
                showlegend=True
            )
            
            # Interpretation
            is_significant = pvalue < 0.05
            interpretation = (
                f"{'Significant' if is_significant else 'Not significant'} difference found "
                f"(p = {pvalue:.4f}). "
            )
            
            if abs(cohens_d) < 0.2:
                effect = "negligible"
            elif abs(cohens_d) < 0.5:
                effect = "small"
            elif abs(cohens_d) < 0.8:
                effect = "medium"
            else:
                effect = "large"
            
            interpretation += f"Effect size is {effect} (Cohen's d = {cohens_d:.3f})."
            
            return AnalyticsResult(
                analysis_type='ttest',
                success=True,
                data={
                    'test': test_name,
                    'test_type': test_type,
                    'groups': [str(g) for g in groups],
                    'value_column': value_col,
                    'statistics': {
                        't_statistic': float(statistic),
                        'p_value': float(pvalue),
                        'cohens_d': float(cohens_d),
                        'is_significant': is_significant,
                        'alpha': 0.05
                    },
                    'group_statistics': {
                        str(groups[0]): {
                            'mean': float(group1_data.mean()),
                            'std': float(group1_data.std()),
                            'n': len(group1_data)
                        },
                        str(groups[1]): {
                            'mean': float(group2_data.mean()),
                            'std': float(group2_data.std()),
                            'n': len(group2_data)
                        }
                    },
                    'interpretation': interpretation
                },
                visualization=fig.to_dict(),
                message=interpretation
            )
            
        except Exception as e:
            logger.error(f"T-test error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='ttest',
                success=False,
                data={},
                error=str(e),
                message=f"T-test failed: {str(e)}"
            )
    
    def anova_test(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str
    ) -> AnalyticsResult:
        """Perform One-Way ANOVA test."""
        try:
            # Get groups
            groups = df[group_col].unique()
            
            if len(groups) < 2:
                return AnalyticsResult(
                    analysis_type='anova',
                    success=False,
                    data={},
                    error="ANOVA requires at least 2 groups",
                    message="ANOVA requires at least 2 groups"
                )
            
            # Prepare group data
            group_data = [df[df[group_col] == g][value_col].dropna() for g in groups]
            
            # Perform ANOVA
            statistic, pvalue = stats.f_oneway(*group_data)
            
            # Calculate group statistics
            group_stats = {}
            for i, g in enumerate(groups):
                data = group_data[i]
                group_stats[str(g)] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'n': len(data),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
            
            # Visualization - Box plot for all groups
            fig = go.Figure()
            for g in groups:
                data = df[df[group_col] == g][value_col].dropna()
                fig.add_trace(go.Box(y=data, name=str(g)))
            
            fig.update_layout(
                title=f'One-Way ANOVA: {value_col} across {group_col}',
                yaxis_title=value_col,
                xaxis_title=group_col
            )
            
            # Interpretation
            is_significant = pvalue < 0.05
            interpretation = (
                f"{'Significant' if is_significant else 'Not significant'} differences found "
                f"among groups (F = {statistic:.4f}, p = {pvalue:.4f})."
            )
            
            return AnalyticsResult(
                analysis_type='anova',
                success=True,
                data={
                    'test': 'One-Way ANOVA',
                    'groups': [str(g) for g in groups],
                    'n_groups': len(groups),
                    'value_column': value_col,
                    'statistics': {
                        'f_statistic': float(statistic),
                        'p_value': float(pvalue),
                        'is_significant': is_significant,
                        'alpha': 0.05
                    },
                    'group_statistics': group_stats,
                    'interpretation': interpretation
                },
                visualization=fig.to_dict(),
                message=interpretation
            )
            
        except Exception as e:
            logger.error(f"ANOVA error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='anova',
                success=False,
                data={},
                error=str(e),
                message=f"ANOVA failed: {str(e)}"
            )
    
    def chi_square_test(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str
    ) -> AnalyticsResult:
        """Perform Chi-Square test of independence."""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(df[col1], df[col2])
            
            # Perform chi-square test
            chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate Cramér's V (effect size)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim))
            
            # Visualization - Heatmap of contingency table
            fig = go.Figure(data=go.Heatmap(
                z=contingency_table.values,
                x=contingency_table.columns.tolist(),
                y=contingency_table.index.tolist(),
                colorscale='Blues',
                text=contingency_table.values,
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title=f'Contingency Table: {col1} vs {col2}',
                xaxis_title=col2,
                yaxis_title=col1
            )
            
            # Interpretation
            is_significant = pvalue < 0.05
            interpretation = (
                f"{'Significant' if is_significant else 'No significant'} association found "
                f"between {col1} and {col2} (χ² = {chi2:.4f}, p = {pvalue:.4f}, "
                f"Cramér's V = {cramers_v:.3f})."
            )
            
            return AnalyticsResult(
                analysis_type='chi_square',
                success=True,
                data={
                    'test': 'Chi-Square Test of Independence',
                    'variables': [col1, col2],
                    'statistics': {
                        'chi2_statistic': float(chi2),
                        'p_value': float(pvalue),
                        'degrees_of_freedom': int(dof),
                        'cramers_v': float(cramers_v),
                        'is_significant': is_significant,
                        'alpha': 0.05
                    },
                    'contingency_table': contingency_table.to_dict(),
                    'expected_frequencies': pd.DataFrame(expected).to_dict(),
                    'interpretation': interpretation
                },
                visualization=fig.to_dict(),
                message=interpretation
            )
            
        except Exception as e:
            logger.error(f"Chi-square test error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='chi_square',
                success=False,
                data={},
                error=str(e),
                message=f"Chi-square test failed: {str(e)}"
            )
    
    def correlation_analysis(
        self,
        df: pd.DataFrame,
        method: str = 'pearson'
    ) -> AnalyticsResult:
        """Perform correlation analysis on numeric columns."""
        try:
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return AnalyticsResult(
                    analysis_type='correlation',
                    success=False,
                    data={},
                    error="Need at least 2 numeric columns",
                    message="Correlation analysis requires at least 2 numeric columns"
                )
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr(method=method)
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    corr_pairs.append({
                        'var1': numeric_cols[i],
                        'var2': numeric_cols[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
            
            # Sort by absolute correlation
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Visualization - Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=numeric_cols,
                y=numeric_cols,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title='Correlation')
            ))
            
            fig.update_layout(
                title=f'{method.capitalize()} Correlation Matrix',
                xaxis_title='Variables',
                yaxis_title='Variables',
                width=800,
                height=800
            )
            
            return AnalyticsResult(
                analysis_type='correlation',
                success=True,
                data={
                    'method': method,
                    'n_variables': len(numeric_cols),
                    'variables': numeric_cols,
                    'correlation_matrix': corr_matrix.to_dict(),
                    'top_correlations': corr_pairs[:10],  # Top 10
                    'summary': f"Analyzed {len(numeric_cols)} variables using {method} correlation"
                },
                visualization=fig.to_dict(),
                message=f"Correlation analysis completed. Found {len(corr_pairs)} pairwise correlations."
            )
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {e}", exc_info=True)
            return AnalyticsResult(
                analysis_type='correlation',
                success=False,
                data={},
                error=str(e),
                message=f"Correlation analysis failed: {str(e)}"
            )


# Create singleton instance
analytics_service = AdvancedAnalyticsService()
