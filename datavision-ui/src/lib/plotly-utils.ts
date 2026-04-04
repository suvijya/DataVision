// ===========================
// Plotly Utilities
// ===========================

import type { PlotTrace } from '@/types';

/**
 * Decode Plotly binary data format (bdata + dtype).
 */
export function decodeBinaryData(binaryObj: { bdata: string; dtype: string }): number[] {
  if (!binaryObj?.bdata) return [];

  try {
    const binaryString = atob(binaryObj.bdata);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const view = new DataView(bytes.buffer);
    const result: number[] = [];

    const decoders: Record<string, () => void> = {
      f8: () => { for (let i = 0; i < bytes.length; i += 8) result.push(view.getFloat64(i, true)); },
      f4: () => { for (let i = 0; i < bytes.length; i += 4) result.push(view.getFloat32(i, true)); },
      i8: () => { for (let i = 0; i < bytes.length; i += 8) result.push(Number(view.getBigInt64(i, true))); },
      i4: () => { for (let i = 0; i < bytes.length; i += 4) result.push(view.getInt32(i, true)); },
      i2: () => { for (let i = 0; i < bytes.length; i += 2) result.push(view.getInt16(i, true)); },
      i1: () => { for (let i = 0; i < bytes.length; i += 1) result.push(view.getInt8(i)); },
      u4: () => { for (let i = 0; i < bytes.length; i += 4) result.push(view.getUint32(i, true)); },
      u2: () => { for (let i = 0; i < bytes.length; i += 2) result.push(view.getUint16(i, true)); },
      u1: () => { for (let i = 0; i < bytes.length; i++) result.push(bytes[i]); },
    };

    const decoder = decoders[binaryObj.dtype];
    if (decoder) {
      decoder();
      return result;
    }
    console.warn('Unknown dtype:', binaryObj.dtype);
    return [];
  } catch (error) {
    console.error('Error decoding binary data:', error);
    return [];
  }
}

/**
 * Pre-process Plotly trace data: decode binary, fix missing types, clean hovertemplates.
 */
export function processTraces(traces: PlotTrace[]): PlotTrace[] {
  return traces.map((trace) => {
    const processed = { ...trace };

    // Infer type for geo traces
    if (!processed.type && processed.geo && processed.locationmode) {
      if (processed.z !== undefined) processed.type = 'choropleth';
      else if (processed.lat && processed.lon) processed.type = 'scattergeo';
      else if (processed.locations) processed.type = 'scattergeo';
    }

    // Decode binary fields
    const binaryFields = ['values', 'labels', 'x', 'y', 'z'] as const;
    for (const field of binaryFields) {
      const val = processed[field];
      if (val && typeof val === 'object' && 'bdata' in (val as Record<string, unknown>)) {
        (processed as Record<string, unknown>)[field] = decodeBinaryData(val as { bdata: string; dtype: string });
      }
    }

    // Remove problematic hovertemplates
    if (processed.hovertemplate && typeof processed.hovertemplate === 'string' && processed.hovertemplate.includes('customdata')) {
      delete processed.hovertemplate;
    }

    return processed;
  });
}

/**
 * Ensure layout has sensible defaults for maps and large charts.
 */
export function processLayout(layout: Record<string, unknown>): Record<string, unknown> {
  const processed = { ...layout };

  if (!processed.height) processed.height = 600;
  if (!processed.autosize) processed.autosize = true;

  // Fix geo layout
  if (processed.geo && typeof processed.geo === 'object') {
    const geo = { ...(processed.geo as Record<string, unknown>) };
    if (!geo.bgcolor) geo.bgcolor = 'rgb(229, 229, 229)';
    if (!geo.showland) { geo.showland = true; geo.landcolor = 'rgb(243, 243, 243)'; }
    if (!geo.showcountries) { geo.showcountries = true; geo.countrycolor = 'rgb(204, 204, 204)'; }
    processed.geo = geo;
  }

  return processed;
}
