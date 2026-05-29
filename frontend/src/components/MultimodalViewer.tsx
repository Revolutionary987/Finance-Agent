"use client";

import React from "react";
import { Database, FileText, BarChart3, TrendingUp, TrendingDown } from "lucide-react";

export function MultimodalViewer({ data }: { data: string | null }) {
  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center h-full w-full text-zinc-500 space-y-4">
        <div className="relative">
          <div className="absolute inset-0 bg-zinc-800 blur-xl opacity-20 rounded-full"></div>
          <Database className="w-12 h-12 text-zinc-700 relative z-10" />
        </div>
        <p className="text-[15px] font-medium tracking-wide">No Active Documents</p>
        <p className="text-sm text-zinc-600 max-w-sm text-center leading-relaxed">
          Raw extracted tables, charts, and SEC document snippets will appear here during analysis.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full w-full">
      <div className="flex items-center gap-3 p-5 border-b border-zinc-800/50 bg-black/40">
        <FileText className="w-5 h-5 text-zinc-400" />
        <h2 className="text-[13px] font-bold tracking-widest text-zinc-200 uppercase">Multimodal Viewer</h2>
        <div className="ml-auto flex items-center gap-2">
          <div className="px-2.5 py-1.5 rounded bg-zinc-800 border border-zinc-700 flex items-center gap-2 shadow-sm">
            <BarChart3 className="w-3.5 h-3.5 text-zinc-400" />
            <span className="text-xs font-mono text-zinc-300">XBRL Extracted (Q3 2026)</span>
          </div>
        </div>
      </div>
      
      <div className="flex-1 p-8 overflow-y-auto">
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden shadow-2xl">
          <div className="px-6 py-5 border-b border-zinc-800 bg-zinc-950">
            <h3 className="text-base font-semibold text-zinc-100">Consolidated Balance Sheet Mock</h3>
            <p className="text-sm text-zinc-500 mt-1">Extracted from Form 10-K (in millions)</p>
          </div>
          
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-zinc-400 uppercase bg-zinc-950/50 font-mono border-b border-zinc-800">
              <tr>
                <th scope="col" className="px-6 py-4 font-semibold tracking-wider">Line Item</th>
                <th scope="col" className="px-6 py-4 font-semibold tracking-wider text-right">Q3 2026</th>
                <th scope="col" className="px-6 py-4 font-semibold tracking-wider text-right">Q3 2025</th>
                <th scope="col" className="px-6 py-4 font-semibold tracking-wider text-right">YoY</th>
              </tr>
            </thead>
            <tbody className="font-mono divide-y divide-zinc-800/50">
              <tr className="bg-white/5 hover:bg-white/10 transition-colors">
                <td className="px-6 py-4 font-medium text-zinc-200">Total Assets</td>
                <td className="px-6 py-4 text-right text-zinc-300">4,592.1</td>
                <td className="px-6 py-4 text-right text-zinc-400">4,120.5</td>
                <td className="px-6 py-4 text-right text-emerald-400 flex items-center justify-end gap-1"><TrendingUp className="w-3.5 h-3.5" /> +11.4%</td>
              </tr>
              <tr className="hover:bg-white/5 transition-colors">
                <td className="px-6 py-4 font-medium text-zinc-200">Total Liabilities</td>
                <td className="px-6 py-4 text-right text-zinc-300">2,105.8</td>
                <td className="px-6 py-4 text-right text-zinc-400">1,980.2</td>
                <td className="px-6 py-4 text-right text-amber-400 flex items-center justify-end gap-1"><TrendingUp className="w-3.5 h-3.5" /> +6.3%</td>
              </tr>
              <tr className="bg-white/5 hover:bg-white/10 transition-colors">
                <td className="px-6 py-4 font-medium text-zinc-200">Total Revenue</td>
                <td className="px-6 py-4 text-right text-zinc-300">1,245.3</td>
                <td className="px-6 py-4 text-right text-zinc-400">1,310.0</td>
                <td className="px-6 py-4 text-right text-red-400 flex items-center justify-end gap-1"><TrendingDown className="w-3.5 h-3.5" /> -4.9%</td>
              </tr>
              <tr className="hover:bg-white/5 transition-colors border-l-2 border-l-amber-500 bg-amber-950/10">
                <td className="px-6 py-4 font-medium text-amber-500">Unrecognized Anomaly</td>
                <td className="px-6 py-4 text-right text-amber-500 font-bold">12.4</td>
                <td className="px-6 py-4 text-right text-zinc-500">--</td>
                <td className="px-6 py-4 text-right text-amber-500">FLAG</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
