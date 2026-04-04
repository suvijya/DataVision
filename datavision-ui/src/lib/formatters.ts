// ===========================
// Markdown → HTML Formatters
// ===========================

/**
 * Convert a block of Markdown-table-containing text into HTML.
 * Detects pipe-delimited lines, groups them, and renders <table> elements.
 */
export function formatAnalysisText(text: string): string {
  if (!text) return '<p>No content available</p>';

  let formatted = text.trim();

  // Headers
  formatted = formatted.replace(/^### (.*?) ###[ \t]*$/gm, '<h3 class="analysis-header">$1</h3>');
  formatted = formatted.replace(/^### (.*)$/gm, '<h3 class="analysis-header">$1</h3>');
  formatted = formatted.replace(/^## (.*)$/gm, '<h3 class="analysis-header">$1</h3>');

  // Section separators
  formatted = formatted.replace(/^--- (.*?) ---$/gm, '<h4 class="analysis-section">$1</h4>');
  formatted = formatted.replace(/^---$/gm, '<hr class="analysis-divider">');

  const lines = formatted.split('\n');
  let html = '';
  let tableBuffer: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Skip code fences
    if (line.startsWith('```')) continue;

    const isTableLine =
      line.includes('|') && (line.startsWith('|') || line.split('|').length >= 3);

    if (isTableLine) {
      tableBuffer.push(line);
    } else {
      if (tableBuffer.length > 0) {
        html += renderMarkdownTable(tableBuffer);
        tableBuffer = [];
      }

      if (!line) {
        html += '<br>';
      } else if (line.startsWith('<')) {
        html += line;
      } else if (/^[•\-\*]\s*/.test(line)) {
        const bulletText = line.replace(/^[•\-\*]\s*/, '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html += `<div class="analysis-bullet">• ${bulletText}</div>`;
      } else {
        const pText = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html += `<p class="analysis-text">${pText}</p>`;
      }
    }
  }

  if (tableBuffer.length > 0) {
    html += renderMarkdownTable(tableBuffer);
  }

  return html;
}

/**
 * Convert an array of Markdown table lines into an HTML <table>.
 */
export function renderMarkdownTable(tableLines: string[]): string {
  const rows = tableLines.filter((line) => !/^[|\s\-:]+$/.test(line));
  if (rows.length === 0) return '';

  const parseRow = (rowLine: string): string[] =>
    rowLine
      .split('|')
      .map((cell) => cell.trim())
      .filter((cell, index, arr) => {
        if (index === 0 && cell === '') return false;
        if (index === arr.length - 1 && cell === '') return false;
        return true;
      });

  const headers = parseRow(rows[0]);
  let tableHtml = '<div class="analysis-table-container"><table class="analysis-table">';

  tableHtml += '<thead><tr>';
  headers.forEach((h) => {
    tableHtml += `<th>${h}</th>`;
  });
  tableHtml += '</tr></thead><tbody>';

  for (let i = 1; i < rows.length; i++) {
    const cells = parseRow(rows[i]);
    tableHtml += '<tr>';
    for (let j = 0; j < headers.length; j++) {
      tableHtml += `<td>${cells[j] || ''}</td>`;
    }
    tableHtml += '</tr>';
  }

  tableHtml += '</tbody></table></div>';
  return tableHtml;
}
