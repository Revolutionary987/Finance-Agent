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
  type: "info" | "tool" | "warning" | "success"|"error";
};

type ViewMode = "dashboard" | "trace";

export function AegisDashboard() {
  const [messages, setMessages] = useState<Message[]>([
    { id: "1", role: "system", content: "Aegis initialized. Ready for SEC auditing and fraud analysis." }
  ]);
  const [threadId, setThreadId] = useState<string | null>(null);
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

    // 1. Reset traces
    setTraceNodes([]);
    setMessages((prev) => [...prev, { id: Date.now().toString(), role: "user", content: text }]);
    setIsProcessing(true);

    try {
      // 2. Fetch real data from Hugging Face
      const res = await realFetchToFastAPI(text, file);
      
      // 3. Update the Trace Inspector with REAL data
      updateTraceNode({ 
        id: "node_retrieve", 
        name: "✔ Vector Search", 
        status: "success", 
        // Map your real backend documents into the Trace UI format here
        retrievedDocs: res.retrieved_context.map((doc: any, index: number) => ({
             chunk: doc.page_content,
             similarity: doc.metadata?.score || 0.99, // If your DB returns scores
             page: doc.metadata?.page || index + 1
        }))
      });

      // 4. Update the chat window
      setMessages((prev) => [...prev, {
        id: Date.now().toString(),
        role: "ai",
        content: "Processing complete.",
        display: res.message_display,
      }]);

    } catch (error) {
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleApproval = async (approved: boolean) => {
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

    if (!threadId) return;

    try {
      // Send the feedback to FastAPI
      const response = await fetch("https://aicoder35235-aegis-backend.hf.space/app/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: threadId,
          status: approved ? "Yes" : "No",
          feedback: approved ? undefined : "Action rejected by auditor."
        }),
      });

      const data = await response.json();

      if (approved) {
        updateTraceNode({ id: "node_6_exec", name: "▶ Escalation_Execution", status: "running" });
        addTelemetry(`[SSE] Executing requested action against SEC Database...`, "tool");
        setTimeout(() => {
          updateTraceNode({ id: "node_6_exec", name: "▶ Escalation_Execution", status: "success", latency: "650ms", outputs: { report_generated: true } });
          setMessages((prev) => [
            ...prev,
            { id: Date.now().toString(), role: "ai", content: data.final_output || "Variance report generated and escalated successfully." }
          ]);
          addTelemetry(`[SSE] Execution complete.`, "success");
        }, 1000);
      } else {
        updateTraceNode({ id: "node_6_cancel", name: "▶ Re_Route_Handler", status: "success", latency: "100ms", outputs: { cancelled: true } });
        setMessages((prev) => [
          ...prev,
          { id: Date.now().toString(), role: "ai", content: data.final_output || `Action rejected. Awaiting further prompt modifications.` }
        ]);
      }
    } catch (error) {
      addTelemetry(`[SSE] Error sending feedback to backend.`, "error");
    }
  };

  const realFetchToFastAPI = async (input: string, file: File | null) => {
  // 1. Create a FormData object (the correct way to send files)
  const formData = new FormData();
  formData.append("question", input);
  
  if (threadId) {
    formData.append("thread_id", threadId);
  }

  // 2. Append the actual raw file binary to the payload
  if (file) {
    formData.append("file", file);
  }

  // 3. Send it (Notice we REMOVED the "Content-Type" header. 
  // The browser automatically sets it to multipart/form-data when using FormData)
  const response = await fetch("https://aicoder35235-aegis-backend.hf.space/app/call", {
    method: "POST",
    body: formData, 
  });


    if (!response.ok) {
      throw new Error("Backend connection failed");
    }

    const data = await response.json();

    // Save the thread_id so the database remembers us next time
    if (!threadId) {
      setThreadId(data.thread_id);
    }

    // Determine if we need human approval (you can map this to specific output texts later)
    const requires_approval = data.message_display.includes("anomaly") || data.message_display.includes("Drafting");

    return {
      thread_id: data.thread_id,
      message_display: data.message_display,
      requires_approval: requires_approval
    };
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
