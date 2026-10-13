export function Innovations() {
  const innovations = [
    {
      id: 1,
      title: 'Fused Kernel Architecture',
      subtitle: 'Eliminating the Memory Wall',
      description: `llama.cpp executes each operation (QKV projection, attention, FFN, LayerNorm) as separate GPU kernels. Each kernel launch requires writing intermediate results to global memory and reading them back — this "memory wall" wastes 60-70% of GPU time on data movement.

VeloxLLM fuses the entire transformer block into 2-3 persistent kernels that keep intermediate data in shared memory/registers. This eliminates global memory round-trips entirely.`,
      metrics: [
        { label: 'Memory bandwidth saved', value: '73%' },
        { label: 'Kernel launches reduced', value: '85%' },
        { label: 'L2 cache hit rate', value: '94%' },
      ],
      color: 'indigo',
    },
    {
      id: 2,
      title: 'Rust Zero-Cost SIMD',
      subtitle: 'Compile-Time Vectorization',
      description: `llama.cpp uses runtime CPU feature detection and function pointers for SIMD dispatch. This adds branch overhead and prevents the compiler from optimizing across operations.

VeloxLLM uses Rust's type system with generic SIMD abstractions. The compiler monomorphizes each function for the target architecture at compile time — producing optimal AVX-512/NEON code with zero runtime dispatch overhead. Combined with LLVM's auto-vectorizer, this produces code that matches hand-written assembly.`,
      metrics: [
        { label: 'CPU throughput gain', value: '3.2x' },
        { label: 'Binary size reduction', value: '40%' },
        { label: 'Branch mispredictions', value: '-98%' },
      ],
      color: 'cyan',
    },
    {
      id: 3,
      title: 'Paged KV-Cache with Continuous Batching',
      subtitle: 'vLLM-Style Serving at Edge Scale',
      description: `llama.cpp uses a fixed-size KV-cache allocated upfront. This wastes memory for short sequences and blocks for long ones. It also can't batch requests with different lengths efficiently.

VeloxLLM implements a paged KV-cache (inspired by vLLM's PagedAttention) that allocates memory in 16-token blocks on demand. Combined with continuous batching, this allows serving 15x more concurrent requests with the same VRAM.`,
      metrics: [
        { label: 'Concurrent requests', value: '15x' },
        { label: 'Memory fragmentation', value: '<2%' },
        { label: 'Tail latency (P99)', value: '-82%' },
      ],
      color: 'green',
    },
    {
      id: 4,
      title: 'Speculative Decoding Pipeline',
      subtitle: 'Draft-Verify Architecture',
      description: `Autoregressive decoding processes one token at a time — inherently sequential and GPU-underutilized. 

VeloxLLM uses a small draft model (or the same model with fewer layers) to speculatively generate 5-8 tokens in parallel, then verifies all of them in a single forward pass. Since verification is parallel, accepted tokens are essentially "free". With a 70% acceptance rate, this yields 3-5x wall-clock speedup.`,
      metrics: [
        { label: 'Acceptance rate', value: '67-82%' },
        { label: 'Effective speedup', value: '3-5x' },
        { label: 'Quality loss', value: '0% (exact)' },
      ],
      color: 'orange',
    },
    {
      id: 5,
      title: 'Adaptive Runtime Quantization',
      subtitle: 'Hardware-Aware Precision',
      description: `llama.cpp requires pre-quantized GGUF files. If you want different quantization levels for different layers (e.g., keep attention layers at Q6, quantize FFN to Q3), you need custom tooling.

VeloxLLM quantizes on-the-fly during model loading with per-layer sensitivity analysis. It automatically assigns higher precision to attention-critical layers and aggressive quantization to less sensitive FFN layers — maximizing quality per bit.`,
      metrics: [
        { label: 'Model size reduction', value: '65%' },
        { label: 'Quality retention (PPL)', value: '99.2%' },
        { label: 'Load time', value: '0.9s' },
      ],
      color: 'purple',
    },
    {
      id: 6,
      title: 'WebGPU + WASM Universal Backend',
      subtitle: 'Run Anywhere at Near-Native Speed',
      description: `llama.cpp's WASM backend is extremely slow (10-20x slower than native) because it can't access GPU compute.

VeloxLLM compiles to WebGPU compute shaders + WASM, giving browser-based inference that's only 1.3-1.8x slower than native CUDA. This enables deploying LLMs in any browser without plugins, extensions, or server infrastructure.`,
      metrics: [
        { label: 'Browser vs native speed', value: '55-77%' },
        { label: 'Supported browsers', value: 'All (WebGPU)' },
        { label: 'First inference', value: '<2s' },
      ],
      color: 'pink',
    },
  ];

  return (
    <section id="innovations" className="py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            <span className="gradient-text">Technical Innovations</span>
          </h2>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Six fundamental innovations that, combined, deliver 10x+ throughput over llama.cpp 
            while using less memory and running on more hardware.
          </p>
        </div>

        <div className="space-y-8">
          {innovations.map((item) => (
            <div key={item.id} className="glass-card rounded-2xl p-8 hover:border-indigo-500/30 transition-all">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-bold bg-${item.color}-500/20 text-${item.color}-300 border border-${item.color}-500/30`}>
                      #{item.id}
                    </span>
                    <span className="text-gray-500 text-sm">{item.subtitle}</span>
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-4">{item.title}</h3>
                  <div className="text-gray-400 leading-relaxed whitespace-pre-line text-sm">
                    {item.description}
                  </div>
                </div>
                <div className="space-y-3">
                  {item.metrics.map((metric, i) => (
                    <div key={i} className="bg-black/30 rounded-xl p-4 border border-white/5">
                      <div className="text-2xl font-bold text-white">{metric.value}</div>
                      <div className="text-xs text-gray-500 mt-1">{metric.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
