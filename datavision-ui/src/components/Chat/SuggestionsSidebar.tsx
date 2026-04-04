'use client';

import { useSession } from '@/hooks/useSession';

const SUGGESTIONS = {
  exploration: [
    'Show me a summary of the data',
    'What are the data types of each column?',
    'Are there any missing values?',
    'Show the first 10 rows',
    'Count unique values in each column',
  ],
  visualization: [
    'Create a bar chart of the top values',
    'Show a histogram of numeric columns',
    'Create a correlation heatmap',
    'Plot a scatter chart',
    'Show a pie chart of categories',
  ],
  analysis: [
    'What are the correlations between variables?',
    'Perform a statistical summary',
    'Detect outliers in the data',
    'Group by category and show averages',
    'Run a regression analysis',
  ],
};

interface Props {
  onClose: () => void;
}

export default function SuggestionsSidebar({ onClose }: Props) {
  const { sendQuery, sessionId } = useSession();

  if (!sessionId) return null;

  return (
    <div className="suggestions-sidebar" id="suggestionsSidebar">
      <div className="sidebar-header">
        <h4><i className="fas fa-lightbulb" /> Smart Suggestions</h4>
        <button className="close-sidebar" onClick={onClose}>
          <i className="fas fa-times" />
        </button>
      </div>
      <div className="suggestion-categories">
        <div className="category" data-category="exploration">
          <h5><i className="fas fa-search" /> Data Exploration</h5>
          <div className="suggestions-list" id="explorationSuggestions">
            {SUGGESTIONS.exploration.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => sendQuery(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
        <div className="category" data-category="visualization">
          <h5><i className="fas fa-chart-line" /> Visualizations</h5>
          <div className="suggestions-list" id="visualizationSuggestions">
            {SUGGESTIONS.visualization.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => sendQuery(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
        <div className="category" data-category="analysis">
          <h5><i className="fas fa-calculator" /> Statistical Analysis</h5>
          <div className="suggestions-list" id="analysisSuggestions">
            {SUGGESTIONS.analysis.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => sendQuery(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
