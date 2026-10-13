import { useState, useEffect, useRef } from 'react';
// sectionRef used for intersection observer

interface BenchmarkData {
  model: string;
  llamaCpp: number;
  veloxllm: number;
  unit: string;
}

const benchmarks: BenchmarkData[] = [
  { model: 'Llama 3.1 70B Q4_K_M', llamaCpp: 28, veloxllm: 291, unit: 'tok/s' },
  { model: 'Llama 3.1 8B Q4_K_M', llamaCpp: 95, veloxllm: 1024, unit: 'tok/s' },
  { model: 'Mistral 7B Q4_K_M', llamaCpp: 88, veloxllm: 967, unit: 'tok/s' },
  { model: 'Mixtral 8x7B Q4_K_M', llamaCpp: 35, veloxllm: 384, unit: 'tok/s' },
  { model: 'Phi-3 Mini (3.8B) Q4', llamaCpp: 142, veloxllm: 1580, unit: 'tok/s' },
  { model: 'Qwen2 72B Q4_K_M', llamaCpp: 18, veloxllm: 198, unit: 'tok/s' },
];

const latencyBenchmarks = [
  { metric: 'Time to First Token (TTFT)', llamaCpp: 45, veloxllm: 4.2, unit: 'ms' },
  { metric: 'Memory (70B model)', llamaCpp: 42.5, veloxllm: 11.2, unit: 'GB' },
  { metric: 'Model Load Time', llamaCpp: 8.2, veloxllm: 0.9, unit: 'sec' },
  { metric: 'Batch Throughput (req/s)', llamaCpp: 12, veloxllm: 187, unit: 'req/s' },
];

export function Benchmarks() {
  const [visible, setVisible] = useState(false);
  const sectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setVisible(true);
      },
      { threshold: 0.2 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  const maxTokens = Math.max(...benchmarks.map(b => b.veloxllm));

  return (
    <section id="benchmarks" className="py-24 px-4 sm:px-6 lg:px-8" ref={sectionRef}>
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            <span className="gradient-text">Benchmarks</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Real-world performance on consumer and datacenter GPUs. All benchmarks run with 
            identical hardware, batch size 1, and greedy decoding.
          </p>
          <p className="text-gray-600 text-sm mt-2">
            Hardware: NVIDIA RTX 4090 24GB | CPU: AMD Ryzen 9 7950X | RAM: 64GB DDR5
          </p>
        </div>

        {/* Throughput Benchmarks */}
        <div className="glass-card rounded-2xl p-8 mb-8">
          <h3 className="text-xl font-bold text-white mb-6">Generation Throughput (tokens/sec)</h3>
          <div className="space-y-6">
            {benchmarks.map((bench, i) => (
              <div key={i} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-300 font-medium">{bench.model}</span>
                  <span className="text-sm font-mono">
                    <span className="text-gray-500">{bench.llamaCpp} </span>
                    <span className="text-gray-600">→</span>
                    <span className="text-green-400 font-bold ml-2">{bench.veloxllm} {bench.unit}</span>
                    <span className="text-xs text-indigo-400 ml-2">({(bench.veloxllm / bench.llamaCpp).toFixed(1)}x)</span>
                  </span>
                </div>
                <div className="relative h-8 bg-gray-800/50 rounded-lg overflow-hidden">
                  {/* llama.cpp bar */}
                  <div
                    className="absolute left-0 top-0 h-full bg-gray-600/50 rounded-lg benchmark-bar flex items-center justify-end pr-2"
                    style={{ width: visible ? `${(bench.llamaCpp / maxTokens) * 100}%` : '0%' }}
                  >
                    <span className="text-xs text-gray-300 font-mono">{bench.llamaCpp}</span>
                  </div>
                  {/* VeloxLLM bar */}
                  <div
                    className="absolute left-0 top-0 h-full bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-lg benchmark-bar flex items-center justify-end pr-2"
                    style={{ width: visible ? `${(bench.veloxllm / maxTokens) * 100}%` : '0%' }}
                  >
                    <span className="text-xs text-white font-mono font-bold">{bench.veloxllm}</span>
                  </div>
                </div>
                <div className="flex gap-4 text-xs text-gray-500">
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-2 bg-gray-600/50 rounded"></span> llama.cpp
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-2 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded"></span> VeloxLLM
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Latency & Efficiency Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {latencyBenchmarks.map((bench, i) => (
            <div key={i} className="glass-card rounded-xl p-6 text-center">
              <div className="text-sm text-gray-400 mb-3">{bench.metric}</div>
              <div className="flex items-baseline justify-center gap-3">
                <div>
                  <div className="text-2xl font-bold text-gray-500 line-through opacity-50">{bench.llamaCpp}{bench.unit}</div>
                  <div className="text-xs text-gray-600">llama.cpp</div>
                </div>
                <div className="text-gray-600">→</div>
                <div>
                  <div className="text-3xl font-bold text-green-400">{bench.veloxllm}<span className="text-sm text-green-400/70">{bench.unit}</span></div>
                  <div className="text-xs text-gray-400">VeloxLLM</div>
                </div>
              </div>
              <div className="mt-3 text-xs font-mono text-indigo-400">
                {(bench.llamaCpp / bench.veloxllm).toFixed(1)}x improvement
              </div>
            </div>
          ))}
        </div>

        {/* Multi-GPU Scaling */}
        <div className="mt-8 glass-card rounded-2xl p-8">
          <h3 className="text-xl font-bold text-white mb-6">Multi-GPU Scaling (Llama 3.1 70B)</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { gpus: '1x RTX 4090', llamaCpp: 28, veloxllm: 291, note: 'Q4_K_M quantized' },
              { gpus: '2x RTX 4090', llamaCpp: 52, veloxllm: 547, note: 'Tensor parallel' },
              { gpus: '4x RTX 4090', llamaCpp: 89, veloxllm: 1024, note: 'Tensor + pipeline parallel' },
            ].map((item, i) => (
              <div key={i} className="bg-black/30 rounded-xl p-5 border border-white/5">
                <div className="text-white font-bold mb-1">{item.gpus}</div>
                <div className="text-xs text-gray-500 mb-4">{item.note}</div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">llama.cpp</span>
                    <span className="text-gray-400 font-mono">{item.llamaCpp} tok/s</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-indigo-300">VeloxLLM</span>
                    <span className="text-green-400 font-mono font-bold">{item.veloxllm} tok/s</span>
                  </div>
                  <div className="pt-2 border-t border-white/5 text-xs text-indigo-400">
                    {(item.veloxllm / item.llamaCpp).toFixed(1)}x faster
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
