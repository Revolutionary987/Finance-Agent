"use client";

import React, { useState } from "react";
import { CheckCircle2, CircleDashed, AlertTriangle, XCircle, Clock, ChevronRight, ChevronDown, FileText, Share2, Server } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type TraceNodeStatus = "pending" | "running" | "success" | "warning" | "error";

export interface TraceNode {
  id: string;
  name: string;
  status: TraceNodeStatus;
  latency?: string;
  inputs?: any;
  outputs?: any;
  retrievedDocs?: { chunk: string; similarity: number; page: number }[];
  routingRationale?: string;
}

interface TraceInspectorProps {
  nodes: TraceNode[];
}

export function TraceInspector({ nodes }: TraceInspectorProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    inputs: true,
    outputs: true,
    docs: true,
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const getStatusIcon = (status: TraceNodeStatus) => {
    switch (status) {
      case "pending": return <CircleDashed className="w-4 h-4 text-zinc-600" />;
      case "running": return <CircleDashed className="w-4 h-4 text-emerald-400 animate-spin" />;
      case "success": return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
      case "warning": return <AlertTriangle className="w-4 h-4 text-amber-500" />;
      case "error": return <XCircle className="w-4 h-4 text-red-500" />;
    }
  };

  const selectedNode = nodes.find(n => n.id === selectedNodeId) || nodes[nodes.length - 1];

  return (
    <div className="flex h-full w-full bg-zinc-950 font-sans text-zinc-200 divide-x divide-zinc-800">
      
      {/* Left Sidebar: Execution Tree */}
      <div className="w-[35%] flex flex-col h-full bg-black/40">
        <div className="p-4 border-b border-zinc-800 flex items-center gap-2 bg-zinc-900/50">
          <Server className="w-4 h-4 text-zinc-400" />
          <h2 className="text-xs font-semibold tracking-wider text-zinc-300 uppercase">Execution Trace</h2>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-1">
          {nodes.map((node, i) => {
            const isSelected = (selectedNodeId === node.id) || (!selectedNodeId && i === nodes.length - 1);
            return (
              <button
                key={node.id}
                onClick={() => setSelectedNodeId(node.id)}
                className={cn(
                  "w-full text-left px-3 py-3 rounded-lg flex flex-col gap-2 transition-all border",
                  isSelected 
                    ? "bg-zinc-800/80 border-zinc-700 shadow-sm" 
                    : "bg-transparent border-transparent hover:bg-zinc-900"
                )}
              >
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(node.status)}
                    <span className={cn(
                      "text-sm font-mono tracking-tight", 
                      isSelected ? "text-zinc-100 font-semibold" : "text-zinc-400"
                    )}>
                      {node.name}
                    </span>
                  </div>
                  {node.latency && (
                    <div className="flex items-center gap-1 text-zinc-500 text-[10px] font-mono bg-black/50 px-1.5 py-0.5 rounded">
                      <Clock className="w-3 h-3" />
                      {node.latency}
                    </div>
                  )}
                </div>
              </button>
            );
          })}
          {nodes.length === 0 && (
            <div className="text-zinc-500 text-xs text-center py-6 font-mono">Awaiting execution...</div>
          )}
        </div>
      </div>

      {/* Main Workspace: Details Panel */}
      <div className="flex-1 flex flex-col h-full bg-zinc-950">
        {selectedNode ? (
          <>
            {/* Header */}
            <div className="p-5 border-b border-zinc-800 bg-zinc-900/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-zinc-900 rounded-lg border border-zinc-800 shadow-inner">
                  {getStatusIcon(selectedNode.status)}
                </div>
                <div>
                  <h3 className="text-lg font-mono font-bold text-zinc-100 tracking-tight">{selectedNode.name}</h3>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">Node Inspector</span>
                    {selectedNode.latency && (
                      <span className="text-[11px] text-zinc-400 font-mono bg-zinc-800/50 px-2 py-0.5 rounded-full border border-zinc-700/50">
                        Duration: {selectedNode.latency}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Content Scroll Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              
              {/* Routing Rationale */}
              {selectedNode.routingRationale && (
                <div className="bg-amber-950/20 border border-amber-900/30 rounded-xl p-4 shadow-sm">
                  <div className="flex items-center gap-2 mb-2 text-amber-500">
                    <Share2 className="w-4 h-4" />
                    <h4 className="text-xs font-bold uppercase tracking-widest">Routing Rationale</h4>
                  </div>
                  <p className="text-sm text-zinc-300 leading-relaxed font-mono">
                    {selectedNode.routingRationale}
                  </p>
                </div>
              )}

              {/* Inputs */}
              <div className="border border-zinc-800 rounded-xl overflow-hidden bg-black/40">
                <button 
                  onClick={() => toggleSection('inputs')}
                  className="w-full flex items-center justify-between p-3 bg-zinc-900/80 hover:bg-zinc-800 transition-colors border-b border-zinc-800"
                >
                  <span className="text-xs font-bold tracking-widest text-zinc-400 uppercase">Inputs</span>
                  {expandedSections.inputs ? <ChevronDown className="w-4 h-4 text-zinc-500" /> : <ChevronRight className="w-4 h-4 text-zinc-500" />}
                </button>
                {expandedSections.inputs && (
                  <div className="p-4 bg-zinc-950 overflow-x-auto">
                    <pre className="text-xs font-mono text-zinc-300 leading-relaxed">
                      {selectedNode.inputs ? JSON.stringify(selectedNode.inputs, null, 2) : "{}"}
                    </pre>
                  </div>
                )}
              </div>

              {/* Outputs */}
              {selectedNode.outputs && (
                <div className="border border-zinc-800 rounded-xl overflow-hidden bg-black/40">
                  <button 
                    onClick={() => toggleSection('outputs')}
                    className="w-full flex items-center justify-between p-3 bg-zinc-900/80 hover:bg-zinc-800 transition-colors border-b border-zinc-800"
                  >
                    <span className="text-xs font-bold tracking-widest text-zinc-400 uppercase">Outputs</span>
                    {expandedSections.outputs ? <ChevronDown className="w-4 h-4 text-zinc-500" /> : <ChevronRight className="w-4 h-4 text-zinc-500" />}
                  </button>
                  {expandedSections.outputs && (
                    <div className="p-4 bg-zinc-950 overflow-x-auto">
                      <pre className="text-xs font-mono text-emerald-400/90 leading-relaxed">
                        {JSON.stringify(selectedNode.outputs, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}

              {/* Retrieved Documents */}
              {selectedNode.retrievedDocs && selectedNode.retrievedDocs.length > 0 && (
                <div className="border border-zinc-800 rounded-xl overflow-hidden bg-black/40">
                  <button 
                    onClick={() => toggleSection('docs')}
                    className="w-full flex items-center justify-between p-3 bg-zinc-900/80 hover:bg-zinc-800 transition-colors border-b border-zinc-800"
                  >
                    <span className="text-xs font-bold tracking-widest text-zinc-400 uppercase flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Retrieved Context ({selectedNode.retrievedDocs.length})
                    </span>
                    {expandedSections.docs ? <ChevronDown className="w-4 h-4 text-zinc-500" /> : <ChevronRight className="w-4 h-4 text-zinc-500" />}
                  </button>
                  {expandedSections.docs && (
                    <div className="p-4 space-y-4">
                      {selectedNode.retrievedDocs.map((doc, idx) => (
                        <div key={idx} className="bg-zinc-900/50 border border-zinc-800/80 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-3 border-b border-zinc-800 pb-2">
                            <span className="text-[10px] font-mono text-emerald-500 bg-emerald-500/10 px-2 py-1 rounded">Score: {doc.similarity.toFixed(4)}</span>
                            <span className="text-[10px] font-mono text-zinc-500 bg-black px-2 py-1 rounded">Page {doc.page}</span>
                          </div>
                          <p className="text-xs font-mono text-zinc-300 leading-relaxed">
                            {doc.chunk}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-zinc-600 text-sm font-mono">
            Select a node to inspect trace details
          </div>
        )}
      </div>
    </div>
  );
}
