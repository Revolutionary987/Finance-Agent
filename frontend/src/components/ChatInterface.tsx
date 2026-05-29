"use client";

import React, { useState, useRef, useEffect } from "react";
import { Message } from "./AegisDashboard";
import { Send, Loader2, Check, X, ShieldAlert, AlertTriangle, Paperclip, File as FileIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface ChatInterfaceProps {
  messages: Message[];
  onSendMessage: (msg: string, file: File | null) => void;
  isProcessing: boolean;
  isAwaitingFeedback: boolean;
  onFeedback: (approved: boolean) => void;
}

export function ChatInterface({ messages, onSendMessage, isProcessing, isAwaitingFeedback, onFeedback }: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const [stagedFile, setStagedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isAwaitingFeedback, isProcessing, stagedFile]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((input.trim() || stagedFile) && !isProcessing && !isAwaitingFeedback) {
      onSendMessage(input, stagedFile);
      setInput("");
      setStagedFile(null);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setStagedFile(file);
    }
  };

  return (
    <div className="flex flex-col h-full w-full">
      {/* Header */}
      <div className="p-5 border-b border-zinc-800/60 bg-black/60 backdrop-blur-md flex items-center gap-3">
        <div className="h-8 w-8 rounded-full bg-zinc-800 flex items-center justify-center border border-zinc-700 shadow-inner">
          <ShieldAlert className="w-4 h-4 text-zinc-300" />
        </div>
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-zinc-100">Aegis AI</h1>
          <p className="text-xs text-zinc-500 font-medium uppercase tracking-widest">Audit Terminal Active</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-8 scroll-smooth" ref={scrollRef}>
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={cn(
              "flex flex-col max-w-[85%] space-y-2",
              msg.role === "user" ? "ml-auto items-end" : "mr-auto items-start"
            )}
          >
            {msg.role !== "user" && (
              <span className="text-[10px] uppercase tracking-wider text-zinc-500 font-bold ml-1">
                {msg.role === "system" ? "System" : "Aegis AI"}
              </span>
            )}
            <div
              className={cn(
                "px-5 py-4 rounded-2xl text-[15px] leading-relaxed shadow-sm",
                msg.role === "user"
                  ? "bg-zinc-100 text-zinc-900 rounded-br-sm font-medium"
                  : msg.role === "system"
                  ? "bg-zinc-900 border border-zinc-800 text-zinc-400 rounded-bl-sm font-mono text-sm"
                  : "bg-zinc-800/80 border border-zinc-700/50 text-zinc-200 rounded-bl-sm backdrop-blur-sm"
              )}
            >
              {msg.display || msg.content}
            </div>
          </div>
        ))}

        {isProcessing && (
          <div className="mr-auto max-w-[80%] flex items-center gap-3 px-5 py-4 rounded-2xl bg-zinc-800/30 border border-zinc-700/30 rounded-bl-sm">
            <Loader2 className="w-4 h-4 text-zinc-400 animate-spin" />
            <span className="text-sm text-zinc-400 font-medium">Processing request...</span>
          </div>
        )}

        {isAwaitingFeedback && (
          <div className="mr-auto w-full bg-zinc-950 border-2 border-amber-500/80 rounded-xl overflow-hidden shadow-[0_0_30px_rgba(245,158,11,0.15)] mt-6">
            <div className="bg-amber-500/10 border-b border-amber-500/30 px-5 py-3 flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-500" />
              <h3 className="text-sm font-bold text-amber-500 tracking-wider uppercase">
                System Handoff: HITL Interrupt
              </h3>
            </div>
            <div className="p-5">
              <div className="mb-5 space-y-3">
                <div className="bg-zinc-900/50 p-4 rounded-lg border border-zinc-800">
                  <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-1">Compiled Context</h4>
                  <p className="text-sm text-zinc-300">
                    Extracted 4 relevant chunks from 10-K detailing revenue recognition. Bear Analyst identified a 14% variance discrepancy in Subscription Services.
                  </p>
                </div>
                <div className="bg-zinc-900/50 p-4 rounded-lg border border-zinc-800">
                  <h4 className="text-xs font-bold text-emerald-500/80 uppercase tracking-widest mb-1">Recommended Action</h4>
                  <p className="text-sm text-emerald-400 font-medium">
                    Flag anomaly, escalate to Auditor Review Queue, and generate formal variance report.
                  </p>
                </div>
              </div>
              <div className="flex gap-4">
                <Button
                  onClick={() => onFeedback(true)}
                  className="flex-1 bg-zinc-100 hover:bg-white text-zinc-900 font-bold shadow-sm h-12 text-sm"
                >
                  <Check className="w-5 h-5 mr-2" /> Approve & Continue
                </Button>
                <Button
                  onClick={() => onFeedback(false)}
                  variant="destructive"
                  className="flex-1 bg-red-950/40 hover:bg-red-900/60 text-red-400 border border-red-900/50 font-bold shadow-sm h-12 text-sm"
                >
                  <X className="w-5 h-5 mr-2" /> Reject / Modify Prompt
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className={cn("p-5 border-t border-zinc-800/60 bg-black/60 backdrop-blur-md transition-all duration-300", isAwaitingFeedback && "opacity-30 grayscale pointer-events-none")}>
        
        {/* Staged File Chip */}
        {stagedFile && (
          <div className="mb-3 flex items-center gap-2 bg-zinc-900 border border-zinc-700/80 w-fit px-3 py-1.5 rounded-lg text-sm text-zinc-200 shadow-sm">
            <FileIcon className="w-4 h-4 text-emerald-500" />
            <span className="max-w-[200px] truncate font-medium">{stagedFile.name}</span>
            <button
              onClick={() => setStagedFile(null)}
              className="ml-2 text-zinc-400 hover:text-red-400 transition-colors bg-zinc-800 p-1 rounded-full"
              title="Remove attachment"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit} className="relative flex items-center">
          {/* Internal Attachment Button */}
          <div className="absolute left-2 z-10">
            <input
              type="file"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf,.csv,image/*"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessing || isAwaitingFeedback}
              className="p-2.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors disabled:opacity-50"
              title="Attach File"
            >
              <Paperclip className="w-5 h-5" />
            </button>
          </div>

          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isProcessing || isAwaitingFeedback}
            placeholder={
              isAwaitingFeedback
                ? "Awaiting your approval..."
                : isProcessing
                ? "Aegis is processing..."
                : "Type your instruction..."
            }
            className="w-full bg-zinc-900/80 border border-zinc-700/50 rounded-xl py-4 pl-14 pr-14 text-[15px] text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500 transition-all disabled:opacity-50 shadow-inner"
          />
          
          <button
            type="submit"
            disabled={(!input.trim() && !stagedFile) || isProcessing || isAwaitingFeedback}
            className="absolute right-2 p-2.5 rounded-lg bg-zinc-100 text-zinc-900 hover:bg-white disabled:opacity-50 disabled:bg-zinc-800 disabled:text-zinc-500 transition-colors"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
        <p className="text-center mt-3 text-xs text-zinc-600 font-medium tracking-wide">SEC 10-K Auditing Agent • End-to-End Encrypted</p>
      </div>
    </div>
  );
}
