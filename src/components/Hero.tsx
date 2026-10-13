import { useEffect, useState } from 'react';

export function Hero() {
  const [tokensPerSec, setTokensPerSec] = useState(0);
  const targetTokens = 2847;

  useEffect(() => {
    const duration = 2000;
    const startTime = Date.now();
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setTokensPerSec(Math.floor(eased * targetTokens));
      if (progress < 1) requestAnimationFrame(animate);
    };
    const timer = setTimeout(animate, 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center hero-gradient overflow-hidden">
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-indigo-400/30 rounded-full float-animation"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 6}s`,
              animationDuration: `${4 + Math.random() * 4}s`,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center pt-20">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-8">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
          <span className="text-sm text-gray-300">Written in Rust + Custom CUDA Kernels + WebGPU</span>
        </div>

        {/* Main heading */}
        <h1 className="text-5xl sm:text-7xl lg:text-8xl font-black tracking-tight mb-6">
          <span className="gradient-text">VeloxLLM</span>
        </h1>
        <p className="text-xl sm:text-2xl text-gray-300 max-w-3xl mx-auto mb-4 leading-relaxed">
          The fastest open-source LLM inference engine on the planet.
        </p>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto mb-10">
          10x faster than llama.cpp through fused kernel architecture, zero-copy memory management, 
          adaptive quantization, and hardware-native SIMD utilization.
        </p>

        {/* Performance stat */}
        <div className="inline-flex items-center gap-6 glass-card rounded-2xl px-8 py-6 mb-10 pulse-glow">
          <div className="text-left">
            <div className="text-4xl sm:text-5xl font-mono font-bold text-white">
              {tokensPerSec.toLocaleString()}
            </div>
            <div className="text-sm text-gray-400 mt-1">tokens/sec on RTX 4090 (Llama 3.1 70B Q4)</div>
          </div>
          <div className="w-px h-16 bg-gray-700"></div>
          <div className="text-left">
            <div className="text-2xl font-bold text-green-400">10.4x</div>
            <div className="text-sm text-gray-400">faster than llama.cpp</div>
          </div>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
          <a href="#readme" className="px-8 py-4 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 text-white font-semibold text-lg hover:opacity-90 transition-all hover:scale-105 shadow-lg shadow-indigo-500/25">
            Quick Start Guide →
          </a>
          <a href="#architecture" className="px-8 py-4 rounded-xl glass-card text-gray-300 font-semibold text-lg hover:text-white transition-all hover:border-indigo-500/50">
            View Architecture
          </a>
        </div>

        {/* Quick stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-3xl mx-auto">
          {[
            { label: 'Memory Usage', value: '-73%', desc: 'vs llama.cpp' },
            { label: 'First Token', value: '<5ms', desc: 'latency' },
            { label: 'GPU Support', value: 'NVIDIA/AMD/Intel', desc: '+ Apple Silicon' },
            { label: 'Languages', value: 'Rust + CUDA', desc: '+ WebGPU WASM' },
          ].map((stat, i) => (
            <div key={i} className="glass-card rounded-xl p-4 text-center">
              <div className="text-lg font-bold text-white">{stat.value}</div>
              <div className="text-xs text-gray-400 mt-1">{stat.label}</div>
              <div className="text-xs text-gray-600">{stat.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
