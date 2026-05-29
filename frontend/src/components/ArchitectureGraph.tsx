"use client";

import React from "react";
import { motion } from "framer-motion";
import { Database, User, Scale, Hand, FileOutput } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export type ActiveNode = "idle" | "upload" | "retriever" | "evaluator" | "hitl" | "output";

interface ArchitectureGraphProps {
  activeNode: ActiveNode;
}

export function ArchitectureGraph({ activeNode }: ArchitectureGraphProps) {
  const nodes = [
    { id: "upload", label: "User Query / Upload", icon: User },
    { id: "retriever", label: "Hybrid Retriever", icon: Database },
    { id: "evaluator", label: "Bull / Bear Evaluator", icon: Scale },
    { id: "hitl", label: "HITL Loop", icon: Hand },
    { id: "output", label: "Final Output", icon: FileOutput },
  ];

  return (
    <div className="flex flex-col h-full w-full bg-zinc-950">
      <div className="flex items-center gap-3 p-4 border-b border-zinc-800/80 bg-black/40">
        <h2 className="text-[11px] font-bold tracking-widest text-zinc-400 uppercase font-sans">
          Execution Topology Minimap
        </h2>
      </div>
      <div className="flex-1 p-6 flex flex-col justify-center items-center relative overflow-hidden">
        {/* Dynamic subtle background glow based on active state */}
        <div 
          className={cn(
            "absolute inset-0 bg-gradient-to-b opacity-5 transition-colors duration-1000",
            activeNode === "hitl" ? "from-amber-500/20 to-transparent" : "from-emerald-500/20 to-transparent"
          )} 
        />
        
        <div className="flex justify-between items-center w-full max-w-2xl relative z-10">
          {nodes.map((node, index) => {
            const isActive = activeNode === node.id;
            const isHitl = node.id === "hitl";
            const Icon = node.icon;
            
            return (
              <React.Fragment key={node.id}>
                {/* The Node */}
                <div className="flex flex-col items-center gap-3 z-10 relative">
                  <motion.div
                    animate={{
                      scale: isActive ? 1.1 : 1,
                      boxShadow: isActive 
                        ? (isHitl ? "0 0 20px rgba(245, 158, 11, 0.4)" : "0 0 20px rgba(16, 185, 129, 0.4)")
                        : "0 0 0px rgba(0,0,0,0)",
                      borderColor: isActive 
                        ? (isHitl ? "rgba(245, 158, 11, 0.8)" : "rgba(16, 185, 129, 0.8)")
                        : "rgba(39, 39, 42, 1)"
                    }}
                    transition={{ duration: 0.3 }}
                    className={cn(
                      "w-12 h-12 rounded-xl flex items-center justify-center border-2 bg-zinc-900 transition-colors",
                      isActive 
                        ? (isHitl ? "bg-amber-950 text-amber-400" : "bg-emerald-950 text-emerald-400")
                        : "text-zinc-500"
                    )}
                  >
                    <Icon className="w-5 h-5" />
                  </motion.div>
                  <div className="text-[10px] uppercase tracking-widest font-mono font-semibold w-24 text-center">
                    <span className={isActive ? (isHitl ? "text-amber-500" : "text-emerald-400") : "text-zinc-500"}>
                      {node.label}
                    </span>
                  </div>
                </div>

                {/* The Connector Line (Skip after last node) */}
                {index < nodes.length - 1 && (
                  <div className="flex-1 h-[2px] bg-zinc-800 mx-2 relative overflow-hidden -mt-8">
                    {/* Active flowing line effect */}
                    {activeNode !== "idle" && (
                      <motion.div
                        className={cn(
                          "absolute top-0 bottom-0 w-full",
                          activeNode === "hitl" ? "bg-amber-500/50" : "bg-emerald-500/50"
                        )}
                        initial={{ x: "-100%" }}
                        animate={{ x: isActive ? "0%" : (nodes.findIndex(n => n.id === activeNode) > index ? "0%" : "-100%") }}
                        transition={{ duration: 0.5 }}
                      />
                    )}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
}
