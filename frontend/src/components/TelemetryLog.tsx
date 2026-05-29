"use client";

import React, { useEffect, useRef } from "react";
import { TelemetryEntry } from "./AegisDashboard";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal, Activity, AlertTriangle, CheckCircle, Search } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function TelemetryLog({ logs }: { logs: TelemetryEntry[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const getIcon = (type: TelemetryEntry["type"]) => {
    switch (type) {
      case "info":
        return <Activity className="w-4 h-4 text-blue-400" />;
      case "tool":
        return <Search className="w-4 h-4 text-purple-400" />;
      case "warning":
        return <AlertTriangle className="w-4 h-4 text-amber-400" />;
      case "success":
        return <CheckCircle className="w-4 h-4 text-emerald-400" />;
    }
  };

  const getColorClass = (type: TelemetryEntry["type"]) => {
    switch (type) {
      case "info":
        return "text-zinc-300";
      case "tool":
        return "text-purple-300 font-medium";
      case "warning":
        return "text-amber-300 font-medium";
      case "success":
        return "text-emerald-300 font-medium";
    }
  };

  return (
    <div className="flex flex-col h-full w-full">
      <div className="flex items-center gap-3 p-4 border-b border-zinc-800/80 bg-zinc-950/80">
        <Terminal className="w-5 h-5 text-zinc-400" />
        <h2 className="text-sm font-bold tracking-widest text-zinc-400 uppercase">Agent Telemetry</h2>
        <div className="ml-auto flex items-center gap-2 px-3 py-1 bg-zinc-900 border border-zinc-800 rounded-full shadow-inner">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="text-xs text-emerald-500/80 font-bold tracking-widest uppercase">Live Stream</span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-5 space-y-3" ref={scrollRef}>
        <AnimatePresence initial={false}>
          {logs.map((log) => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, y: 10, x: -10 }}
              animate={{ opacity: 1, y: 0, x: 0 }}
              className="flex items-start gap-4 text-sm"
            >
              <div className="text-zinc-500 shrink-0 mt-0.5 tracking-tight">
                [{new Date(log.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 })}]
              </div>
              <div className="shrink-0 mt-0.5">{getIcon(log.type)}</div>
              <div className={cn("leading-relaxed", getColorClass(log.type))}>
                {log.text}
              </div>
            </motion.div>
          ))}
          {logs.length === 0 && (
            <div className="text-zinc-600 text-sm italic py-2">Listening for incoming telemetry events...</div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
