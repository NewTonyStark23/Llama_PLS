export function Architecture() {
  return (
    <section id="architecture" className="py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            <span className="gradient-text">Architecture</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            A fundamentally different approach to LLM inference — from the memory allocator to the GPU kernels, 
            every layer is redesigned for maximum throughput.
          </p>
        </div>

        {/* Architecture Diagram */}
        <div className="glass-card rounded-2xl p-8 mb-8 glow-border">
          <h3 className="text-xl font-bold text-white mb-6">System Architecture Overview</h3>
          <div className="rounded-xl overflow-hidden border border-white/10 mb-6">
            <img 
              src="https://image.qwenlm.ai/generated-images/51a225e7-8f44-4936-857c-aded0c2dd8d5/_result.png" 
              alt="VeloxLLM Architecture Diagram"
              className="w-full h-auto"
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Input Layer */}
            <div className="space-y-4">
              <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-xl p-4">
                <h4 className="text-indigo-300 font-semibold mb-2">📥 Input Pipeline</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Zero-copy tokenization (Rust)</li>
                  <li>• Streaming KV-cache prefill</li>
                  <li>• Batch-aware scheduling</li>
                  <li>• Speculative draft tokens</li>
                </ul>
              </div>
              <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4">
                <h4 className="text-cyan-300 font-semibold mb-2">⚡ Compute Engine</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Fused attention+FFN kernels</li>
                  <li>• Persistent thread blocks</li>
                  <li>• Warp-level matrix multiply</li>
                  <li>• Dynamic batch sizing</li>
                </ul>
              </div>
            </div>

            {/* Core Engine */}
            <div className="space-y-4">
              <div className="bg-gradient-to-b from-indigo-500/20 to-cyan-500/20 border border-purple-500/30 rounded-xl p-6 text-center">
                <div className="text-3xl mb-2">🧠</div>
                <h4 className="text-white font-bold text-lg mb-2">VeloxLLM Core</h4>
                <p className="text-sm text-gray-400 mb-4">Rust-based orchestration layer</p>
                <div className="space-y-2 text-left">
                  <div className="bg-black/30 rounded-lg p-2 text-xs font-mono text-green-300">
                    async_compute_graph()
                  </div>
                  <div className="bg-black/30 rounded-lg p-2 text-xs font-mono text-cyan-300">
                    fused_kernel_dispatch()
                  </div>
                  <div className="bg-black/30 rounded-lg p-2 text-xs font-mono text-yellow-300">
                    adaptive_quantize()
                  </div>
                  <div className="bg-black/30 rounded-lg p-2 text-xs font-mono text-purple-300">
                    memory_pool_alloc()
                  </div>
                </div>
              </div>
              <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-4">
                <h4 className="text-purple-300 font-semibold mb-2">🔧 Memory Manager</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Arena allocator (zero-alloc hot path)</li>
                  <li>• Paged KV-cache (vLLM-style)</li>
                  <li>• NUMA-aware allocation</li>
                  <li>• Unified memory (CPU↔GPU)</li>
                </ul>
              </div>
            </div>

            {/* Output Layer */}
            <div className="space-y-4">
              <div className="bg-green-500/10 border border-green-500/30 rounded-xl p-4">
                <h4 className="text-green-300 font-semibold mb-2">🎯 Hardware Backends</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Custom CUDA kernels (SM 8.0+)</li>
                  <li>• AMD ROCm HIP kernels</li>
                  <li>• Apple Metal shaders</li>
                  <li>• WebGPU compute shaders</li>
                  <li>• x86 AVX-512 / ARM NEON</li>
                </ul>
              </div>
              <div className="bg-orange-500/10 border border-orange-500/30 rounded-xl p-4">
                <h4 className="text-orange-300 font-semibold mb-2">📤 Output Pipeline</h4>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>• Speculative verification</li>
                  <li>• Parallel sampling</li>
                  <li>• Beam search (batched)</li>
                  <li>• Streaming detokenization</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Key Design Principles */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            {
              icon: '🔗',
              title: 'Kernel Fusion',
              desc: 'Fuse QKV projection, attention, and FFN into single GPU kernels. Eliminates 73% of memory round-trips that bottleneck llama.cpp.',
              color: 'indigo',
            },
            {
              icon: '🧊',
              title: 'Zero-Copy Memory',
              desc: 'Arena allocator with no heap allocations on the inference hot path. Models are memory-mapped directly from disk with page-level granularity.',
              color: 'cyan',
            },
            {
              icon: '🎯',
              title: 'Adaptive Quantization',
              desc: 'Runtime quantization selection based on available VRAM, compute capability, and accuracy requirements. Supports Q2_K through FP16.',
              color: 'green',
            },
            {
              icon: '⚡',
              title: 'Speculative Decoding',
              desc: 'Use a small draft model to propose 5-8 tokens ahead, then verify in parallel. Achieves 3-5x wall-clock speedup for generation.',
              color: 'orange',
            },
          ].map((item, i) => (
            <div key={i} className={`glass-card rounded-xl p-6 hover:border-${item.color}-500/50 transition-all hover:scale-[1.02]`}>
              <div className="text-3xl mb-3">{item.icon}</div>
              <h4 className="text-white font-bold mb-2">{item.title}</h4>
              <p className="text-sm text-gray-400 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>

        {/* Comparison Table */}
        <div className="mt-16 glass-card rounded-2xl overflow-hidden">
          <div className="p-6 border-b border-white/5">
            <h3 className="text-xl font-bold text-white">Why VeloxLLM vs llama.cpp?</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left px-6 py-4 text-gray-400 font-medium">Feature</th>
                  <th className="text-left px-6 py-4 text-gray-400 font-medium">llama.cpp</th>
                  <th className="text-left px-6 py-4 text-indigo-300 font-medium">VeloxLLM</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Language', 'C/C++', 'Rust + CUDA + WebGPU'],
                  ['Memory Allocator', 'malloc/free per layer', 'Arena allocator (zero-alloc hot path)'],
                  ['Kernel Strategy', 'Separate kernels per op', 'Fused multi-op kernels'],
                  ['Attention', 'Standard / Flash Attn v1', 'Flash Attention v3 + persistent kernels'],
                  ['Quantization', 'Static GGUF formats', 'Adaptive runtime quantization'],
                  ['Batching', 'Sequential / basic batching', 'Continuous batching + paged KV-cache'],
                  ['Speculative Decoding', 'Not implemented', 'Native draft-verify pipeline'],
                  ['CPU SIMD', 'AVX2/AVX512 (runtime)', 'Compile-time SIMD via Rust generics'],
                  ['GPU Backends', 'CUDA, Metal, Vulkan', 'CUDA, ROCm, Metal, WebGPU'],
                  ['Safety', 'C/C++ (UB risks)', 'Rust (memory-safe, no UB)'],
                ].map(([feature, llama, velox], i) => (
                  <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                    <td className="px-6 py-3 text-gray-300 font-medium">{feature}</td>
                    <td className="px-6 py-3 text-gray-500">{llama}</td>
                    <td className="px-6 py-3 text-indigo-200">{velox}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
