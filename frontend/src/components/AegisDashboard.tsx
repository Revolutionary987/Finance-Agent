"use client";

import React, { useState } from "react";
import { ChatInterface } from "@/components/ChatInterface";
import { TraceInspector, TraceNode } from "@/components/TraceInspector";
import { MultimodalViewer } from "@/components/MultimodalViewer";
import { TelemetryLog } from "@/components/TelemetryLog";
import { LayoutDashboard, Microscope } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type Message = {
  id: string;
  role: "user" | "ai" | "system";
  content: string;
  display?: string; 
};

export type TelemetryEntry = {
  id: string;
  timestamp: string;
  text: string;
  type: "info" | "tool" | "warning" | "success";
};

type ViewMode = "dashboard" | "trace";

export function AegisDashboard() {
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", role: "system", content: "Aegis initialized. Ready for SEC auditing and fraud analysis." }
  ]);
  
  // State for Trace view
  const [traceNodes, setTraceNodes] = useState<TraceNode[]>([]);
  
  // State for Dashboard view
  const [telemetry, setTelemetry] = useState<TelemetryEntry[]>([]);
  const [multimodalData, setMultimodalData] = useState<string | null>(null);
  
  // Common state
  const [isAwaitingFeedback, setIsAwaitingFeedback] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // View Toggle State
  const [viewMode, setViewMode] = useState<ViewMode>("dashboard");

  const addTelemetry = (text: string, type: TelemetryEntry["type"] = "info") => {
    setTelemetry((prev) => [
      ...prev,
      {
        id: Math.random().toString(36).substring(7),
        timestamp: new Date().toISOString(),
        text,
        type,
      },
    ]);
  };

  const updateTraceNode = (node: TraceNode) => {
    setTraceNodes(prev => {
      const existingIdx = prev.findIndex(n => n.id === node.id);
      if (existingIdx >= 0) {
        const newNodes = [...prev];
        newNodes[existingIdx] = node;
        return newNodes;
      }
      return [...prev, node];
    });
  };

  const handleSendMessage = async (text: string, file: File | null) => {
    if ((!text.trim() && !file) || isProcessing) return;

    // Reset traces and telemetry for new execution
    setTraceNodes([]);
    setTelemetry([]);
    setMultimodalData(null);

    const userMsg: Message = { 
      id: Date.now().toString(), 
      role: "user", 
      content: file ? `[Attached File: ${file.name}] ${text}` : text 
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsProcessing(true);
    
    // Simulate telemetry
    addTelemetry(`[SSE] Received user input...`, "info");
    if (file) addTelemetry(`[SSE] Processing attached file: ${file.name}`, "tool");
    
    // Simulate trace execution
    const ingestId = "node_1_ingest";
    updateTraceNode({ id: ingestId, name: "▶ Ingestion", status: "running", inputs: { prompt: text, file: file?.name } });
    
    setTimeout(() => {
      updateTraceNode({ id: ingestId, name: "▶ Ingestion", status: "success", latency: "142ms", inputs: { prompt: text, file: file?.name }, outputs: { parsed: true, size: "14KB" } });
      updateTraceNode({ id: "node_2_retrieve", name: "✔ Hybrid_Retriever", status: "running", inputs: { query: text } });
      addTelemetry(`[SSE] Searching SEC database / RAG indices...`, "info");
    }, 600);
    
    setTimeout(() => {
      updateTraceNode({ 
        id: "node_2_retrieve", 
        name: "✔ Hybrid_Retriever", 
        status: "success", 
        latency: "840ms",
        inputs: { query: text },
        outputs: { matches: 4 },
        retrievedDocs: [
          { chunk: "The Company recognized $128.9M in Subscription Services, a significant deviation from prior quarters...", similarity: 0.924, page: 42 },
          { chunk: "Revenue recognition policies shifted in Q2 to account for bundled software licenses...", similarity: 0.881, page: 43 }
        ]
      });
      updateTraceNode({ id: "node_3_eval", name: "⚠ Bear_Analyst_Review", status: "running", inputs: { documents: 2 } });
      addTelemetry(`[SSE] Bull/Bear Debate in progress...`, "tool");
    }, 1500);

    try {
      const res = await mockFetchToFastAPI(text, file);
      
      const aiMsg: Message = {
        id: Date.now().toString(),
        role: "ai",
        content: "Processing complete.",
        display: res.message_display,
      };
      
      if (res.requires_approval) {
        updateTraceNode({ 
          id: "node_3_eval", 
          name: "⚠ Bear_Analyst_Review", 
          status: "success", 
          latency: "1.2s",
          routingRationale: "Reasoning: Bear Analyst flagged a 14% variance discrepancy in Subscription Services -> Routing to HITL Interruption.",
          outputs: { flag: "anomaly_detected", value: "$12.4M" }
        });
        updateTraceNode({ id: "node_4_hitl", name: "⏸ HITL_Interrupt", status: "warning", latency: "Pending", inputs: { await_approval: true } });
        addTelemetry(`[SSE] Halt execution: Awaiting human-in-the-loop approval...`, "warning");
        setIsAwaitingFeedback(true);
      } else {
        updateTraceNode({ 
          id: "node_3_eval", 
          name: "⚠ Bear_Analyst_Review", 
          status: "success", 
          latency: "940ms",
          routingRationale: "Reasoning: No variance detected -> Routing to Generation node.",
          outputs: { flag: "none" }
        });
        updateTraceNode({ id: "node_5_gen", name: "▶ Final_Output_Generator", status: "success", latency: "450ms", outputs: { generated: true } });
        addTelemetry(`[SSE] Generation complete.`, "success");
        setMessages((prev) => [...prev, aiMsg]);
      }
      
    } catch (error) {
      updateTraceNode({ id: "node_error", name: "✖ Error_Handler", status: "error", latency: "0ms" });
      addTelemetry(`[SSE] Error communicating with backend.`, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleApproval = (approved: boolean) => {
    setIsAwaitingFeedback(false);
    addTelemetry(`[SSE] Interrupt resolved: User ${approved ? "approved" : "denied"} the action.`, approved ? "success" : "warning");
    
    updateTraceNode({ 
      id: "node_4_hitl", 
      name: "⏸ HITL_Interrupt", 
      status: "success", 
      latency: "Resolved", 
      inputs: { await_approval: true },
      outputs: { user_action: approved ? "Approved" : "Rejected" } 
    });

    if (approved) {
      updateTraceNode({ id: "node_6_exec", name: "▶ Escalation_Execution", status: "running" });
      addTelemetry(`[SSE] Executing requested action against SEC Database...`, "tool");
      setTimeout(() => {
        updateTraceNode({ id: "node_6_exec", name: "▶ Escalation_Execution", status: "success", latency: "650ms", outputs: { report_generated: true } });
        setMessages((prev) => [
          ...prev,
          { id: Date.now().toString(), role: "ai", content: `Variance report generated and escalated to Auditor Queue successfully.` }
        ]);
        addTelemetry(`[SSE] Execution complete.`, "success");
      }, 1000);
    } else {
      updateTraceNode({ id: "node_6_cancel", name: "▶ Re_Route_Handler", status: "success", latency: "100ms", outputs: { cancelled: true } });
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), role: "ai", content: `Action rejected. Awaiting further prompt modifications.` }
      ]);
    }
  };

  const mockFetchToFastAPI = async (input: string, file: File | null) => {
    return new Promise<{ thread_id: string; message_display: string; requires_approval?: boolean }>((resolve) => {
      if (input.toLowerCase().includes("audit") || (file && file.name.includes("10k"))) {
        setTimeout(() => {
          setMultimodalData("active"); 
          addTelemetry(`[SSE] Detected anomaly in Subscription Services segment.`, "warning");
        }, 2200);
        
        setTimeout(() => {
          resolve({
            thread_id: "thread_x8f9a2",
            message_display: "Anomaly flagged.",
            requires_approval: true
          });
        }, 2500);
      } else {
        setTimeout(() => {
          setMultimodalData(null);
          resolve({
            thread_id: "thread_y9b1c3",
            message_display: "Analysis complete. The financials appear sound with no immediate red flags detected in the latest filings.",
          });
        }, 2200);
      }
    });
  };

  return (
    <div className="flex h-screen w-full bg-[#050505] text-zinc-100 font-sans p-6 gap-6 relative">
      
      {/* Absolute Top Right Toggle */}
      <div className="absolute top-6 right-6 z-50">
        <div className="flex items-center p-1 bg-zinc-950/80 border border-zinc-800 rounded-lg shadow-xl backdrop-blur-xl">
          <button
            onClick={() => setViewMode("dashboard")}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-semibold transition-all",
              viewMode === "dashboard" 
                ? "bg-zinc-800 text-zinc-100 shadow-sm" 
                : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50"
            )}
          >
            <LayoutDashboard className="w-4 h-4" />
            Dashboard View
          </button>
          <button
            onClick={() => setViewMode("trace")}
            className={cn(
              "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-semibold transition-all",
              viewMode === "trace" 
                ? "bg-emerald-950/40 border border-emerald-900/50 text-emerald-400 shadow-sm" 
                : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50"
            )}
          >
            <Microscope className="w-4 h-4" />
            Engine Trace
          </button>
        </div>
      </div>

      {/* Left Column: Chat Interface (40%) */}
      <div className="w-2/5 flex flex-col border border-zinc-800/80 bg-zinc-950/80 rounded-2xl overflow-hidden backdrop-blur-2xl relative z-10 shadow-2xl pt-14">
        {/* PT-14 added above so it aligns with the absolute toggle button on the right if needed, though they are in separate columns. Actually we can just keep chat aligned top. */}
        <div className="absolute top-0 left-0 right-0 h-14 bg-transparent z-0 pointer-events-none" />
        <div className="flex-1 overflow-hidden relative z-10 bg-zinc-950/80">
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            isProcessing={isProcessing}
            isAwaitingFeedback={isAwaitingFeedback}
            onFeedback={handleApproval}
          />
        </div>
      </div>

      {/* Right Column (60%) */}
      <div className="w-3/5 flex flex-col relative rounded-2xl overflow-hidden shadow-2xl pt-14">
        
        {viewMode === "dashboard" && (
          <div className="flex flex-col gap-6 w-full h-full pb-0 relative">
            <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center bg-cover opacity-10 z-0 pointer-events-none rounded-2xl" />
            
            {/* Top: Multimodal Viewer */}
            <div className="flex-1 min-h-0 bg-black/40 border border-zinc-800/60 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-3xl z-10">
              <MultimodalViewer data={multimodalData} />
            </div>
            
            {/* Bottom: Telemetry Log */}
            <div className="h-[40%] min-h-0 bg-black/60 border border-zinc-800/60 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-3xl z-10">
              <TelemetryLog logs={telemetry} />
            </div>
          </div>
        )}

        {viewMode === "trace" && (
          <div className="flex-1 flex flex-col w-full h-full border border-zinc-800/60 rounded-2xl overflow-hidden shadow-2xl bg-black/60 backdrop-blur-3xl relative z-10">
            <TraceInspector nodes={traceNodes} />
          </div>
        )}

      </div>
    </div>
  );
}
