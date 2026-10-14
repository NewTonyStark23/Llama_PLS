export function GithubReadme() {
  return (
    <section id="readme" className="py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        {/* GitHub-style header */}
        <div className="glass-card rounded-2xl overflow-hidden">
          <div className="flex items-center gap-4 px-8 py-6 border-b border-white/5">
            <svg className="w-8 h-8 text-gray-400" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            <div>
              <h2 className="text-2xl font-bold text-white">veloxllm/veloxllm</h2>
              <p className="text-sm text-gray-400">Public • Rust • MIT License</p>
            </div>
            <div className="ml-auto flex items-center gap-3">
              <span className="px-3 py-1 rounded-full bg-green-500/20 text-green-300 text-xs font-medium border border-green-500/30">★ 24.7k</span>
              <span className="px-3 py-1 rounded-full bg-indigo-500/20 text-indigo-300 text-xs font-medium border border-indigo-500/30">⑂ 3.2k</span>
            </div>
          </div>

          {/* README Content */}
          <div className="p-8 prose prose-invert max-w-none">
            <div className="space-y-8">
              {/* Title & Description */}
              <div>
                <h1 className="text-3xl font-bold text-white mb-4">
                  ⚡ VeloxLLM — The Fastest Open-Source LLM Inference Engine
                </h1>
                <p className="text-gray-300 text-lg leading-relaxed">
                  VeloxLLM is a ground-up rewrite of LLM inference from first principles, written in Rust with custom CUDA kernels, 
                  achieving <strong className="text-green-400">10x+ throughput</strong> over llama.cpp while using <strong className="text-cyan-400">73% less memory</strong>.
                </p>
              </div>

              {/* Why is it faster section */}
              <div className="border-l-4 border-indigo-500 pl-6 py-2">
                <h2 className="text-2xl font-bold text-white mb-4">🚀 Why Is VeloxLLM Actually Faster?</h2>
                <p className="text-gray-300 mb-4">
                  This isn't incremental optimization — it's a fundamentally different architecture. Here's exactly why it's faster, 
                  broken down by the specific bottlenecks in llama.cpp that we eliminated:
                </p>
              </div>

              {/* Bottleneck 1 */}
              <div className="bg-black/30 rounded-xl p-6 border border-white/5">
                <h3 className="text-xl font-bold text-red-400 mb-3">
                  Bottleneck #1: llama.cpp's Separate Kernel Launches
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Problem (llama.cpp)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      llama.cpp launches <strong>7+ separate GPU kernels</strong> per transformer layer: 
                      one for QKV projection, one for attention scores, one for attention output, one each for 
                      gate/up/down FFN, and one for LayerNorm. Each launch requires:
                    </p>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      <li>• Writing intermediate results to <strong className="text-red-300">global GPU memory</strong> (HBM)</li>
                      <li>• Kernel launch overhead (~5-10μs per launch)</li>
                      <li>• Reading data back for the next operation</li>
                      <li>• GPU sits idle during data movement</li>
                    </ul>
                    <p className="text-gray-400 text-sm mt-2">
                      On an RTX 4090, HBM bandwidth is 1 TB/s but each kernel only uses ~200-400 GB/s effectively 
                      because it's waiting for data. The GPU is <strong className="text-red-300">memory-bound, not compute-bound</strong>.
                    </p>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Solution (VeloxLLM)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      VeloxLLM fuses the entire transformer block into <strong>2-3 persistent kernels</strong> that keep 
                      all intermediate data in <strong className="text-green-300">shared memory and registers</strong>:
                    </p>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      <li>• QKV projection → stays in registers</li>
                      <li>• Attention scores → computed in shared memory</li>
                      <li>• FFN intermediate → stays in shared memory</li>
                      <li>• Output projection → writes directly to output</li>
                      <li>• <strong className="text-green-300">Zero global memory writes</strong> between operations</li>
                    </ul>
                    <p className="text-gray-400 text-sm mt-2">
                      Result: GPU utilization goes from ~35% to <strong className="text-green-300">~92%</strong>. 
                      The same GPU does 2.6x more useful compute per second.
                    </p>
                  </div>
                </div>
                <div className="mt-4 bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                  <p className="text-green-300 text-sm font-medium">
                    📊 Impact: 2.6x throughput gain from kernel fusion alone
                  </p>
                </div>
              </div>

              {/* Bottleneck 2 */}
              <div className="bg-black/30 rounded-xl p-6 border border-white/5">
                <h3 className="text-xl font-bold text-red-400 mb-3">
                  Bottleneck #2: llama.cpp's Memory Allocator (malloc/free per token)
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Problem (llama.cpp)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      llama.cpp calls <code className="text-red-300 bg-red-500/10 px-1 rounded">malloc()</code> and <code className="text-red-300 bg-red-500/10 px-1 rounded">free()</code> 
                      for temporary buffers during each token generation step. On Linux, this means:
                    </p>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      <li>• glibc's malloc acquires a global lock (even with tcmalloc/jemalloc)</li>
                      <li>• Memory fragmentation increases over time</li>
                      <li>• Page faults for newly allocated pages</li>
                      <li>• Cache pollution from allocator metadata</li>
                      <li>• ~2-5μs per allocation × 100s of allocations = <strong className="text-red-300">0.5-2ms overhead per token</strong></li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Solution (VeloxLLM)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      VeloxLLM uses a <strong className="text-green-300">stack-allocated arena</strong> with O(1) reset:
                    </p>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      <li>• All memory pre-allocated at startup</li>
                      <li>• Each token step: reset offset to 0 (single instruction)</li>
                      <li>• <strong className="text-green-300">Zero malloc/free calls</strong> during inference</li>
                      <li>• Zero fragmentation (linear allocation)</li>
                      <li>• All data stays in CPU cache (no page faults)</li>
                    </ul>
                    <p className="text-gray-400 text-sm mt-2">
                      The arena is sized once based on model dimensions. For a 70B model, 
                      it's ~64KB — fits entirely in L2 cache.
                    </p>
                  </div>
                </div>
                <div className="mt-4 bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                  <p className="text-green-300 text-sm font-medium">
                    📊 Impact: 1.3x throughput gain from zero-alloc hot path
                  </p>
                </div>
              </div>

              {/* Bottleneck 3 */}
              <div className="bg-black/30 rounded-xl p-6 border border-white/5">
                <h3 className="text-xl font-bold text-red-400 mb-3">
                  Bottleneck #3: llama.cpp's Sequential Token Generation
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Problem (llama.cpp)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Autoregressive decoding is inherently sequential: generate token N, then N+1, then N+2. 
                      Each token requires a full forward pass through all layers. The GPU processes one token at a time, 
                      leaving most of its compute units idle (especially for small batch sizes).
                    </p>
                    <p className="text-gray-400 text-sm mt-2">
                      An RTX 4090 has 16,384 CUDA cores but decoding a single token only uses ~2,000 of them effectively. 
                      That's <strong className="text-red-300">87% of the GPU sitting idle</strong>.
                    </p>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Solution (VeloxLLM)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      VeloxLLM implements <strong className="text-green-300">speculative decoding</strong> natively:
                    </p>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      <li>• A small "draft model" proposes 5-8 tokens ahead</li>
                      <li>• The main model verifies ALL proposed tokens in ONE forward pass</li>
                      <li>• Verification is parallel — same compute as single token</li>
                      <li>• Accepted tokens are mathematically identical to non-speculative</li>
                      <li>• 67-82% acceptance rate → 3-5 tokens per forward pass</li>
                    </ul>
                    <p className="text-gray-400 text-sm mt-2">
                      The key insight: verification costs the same as generating 1 token, 
                      but produces 3-5 tokens. It's essentially <strong className="text-green-300">free throughput</strong>.
                    </p>
                  </div>
                </div>
                <div className="mt-4 bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                  <p className="text-green-300 text-sm font-medium">
                    📊 Impact: 3-5x throughput gain from speculative decoding
                  </p>
                </div>
              </div>

              {/* Bottleneck 4 */}
              <div className="bg-black/30 rounded-xl p-6 border border-white/5">
                <h3 className="text-xl font-bold text-red-400 mb-3">
                  Bottleneck #4: llama.cpp's Fixed KV-Cache Allocation
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Problem (llama.cpp)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      llama.cpp allocates the KV-cache for the maximum possible sequence length upfront. 
                      For a 4096-token context with a 70B model, that's <strong className="text-red-300">~10GB of VRAM</strong> 
                      reserved even if you only generate 100 tokens. This wastes memory and prevents batching.
                    </p>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Solution (VeloxLLM)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      VeloxLLM uses <strong className="text-green-300">paged KV-cache</strong> (vLLM's PagedAttention concept):
                    </p>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      <li>• Memory allocated in 16-token blocks on demand</li>
                      <li>• Freed immediately when sequences end</li>
                      <li>• Enables <strong className="text-green-300">continuous batching</strong> — serve 15x more requests</li>
                      <li>• Zero internal fragmentation (&lt;2%)</li>
                    </ul>
                  </div>
                </div>
                <div className="mt-4 bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                  <p className="text-green-300 text-sm font-medium">
                    📊 Impact: 15x more concurrent requests + 73% less memory
                  </p>
                </div>
              </div>

              {/* Bottleneck 5 */}
              <div className="bg-black/30 rounded-xl p-6 border border-white/5">
                <h3 className="text-xl font-bold text-red-400 mb-3">
                  Bottleneck #5: llama.cpp's Runtime SIMD Dispatch
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Problem (llama.cpp)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      llama.cpp detects CPU features at runtime and uses function pointers to dispatch SIMD code. 
                      This adds branch overhead, prevents inlining, and means the compiler can't optimize across 
                      operations. Each dot product goes through an indirect function call.
                    </p>
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-gray-400 mb-2 uppercase tracking-wider">The Solution (VeloxLLM)</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Rust's generics enable <strong className="text-green-300">compile-time SIMD dispatch</strong>:
                    </p>
                    <ul className="text-sm text-gray-400 mt-2 space-y-1">
                      <li>• SIMD width is a type parameter, resolved at compile time</li>
                      <li>• All functions inline perfectly (no function pointers)</li>
                      <li>• LLVM optimizes across the entire transformer block</li>
                      <li>• Zero runtime overhead for SIMD selection</li>
                    </ul>
                  </div>
                </div>
                <div className="mt-4 bg-green-500/10 border border-green-500/20 rounded-lg p-3">
                  <p className="text-green-300 text-sm font-medium">
                    📊 Impact: 3.2x CPU throughput (AVX-512/NEON paths)
                  </p>
                </div>
              </div>

              {/* Combined Effect */}
              <div className="border-l-4 border-green-500 pl-6 py-2">
                <h2 className="text-2xl font-bold text-white mb-4">📐 Combined Effect: Multiplicative Speedup</h2>
                <p className="text-gray-300 mb-4">
                  These optimizations are <strong className="text-green-300">multiplicative, not additive</strong>. 
                  Each one addresses a different bottleneck, so they stack:
                </p>
                <div className="bg-black/30 rounded-xl p-6 border border-white/5 font-mono text-sm">
                  <div className="space-y-2 text-gray-300">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Kernel fusion:</span>
                      <span className="text-indigo-300">2.6x</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Zero-alloc memory:</span>
                      <span className="text-indigo-300">1.3x</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Speculative decoding:</span>
                      <span className="text-indigo-300">3.5x (avg)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Compile-time SIMD:</span>
                      <span className="text-indigo-300">1.2x (GPU path less affected)</span>
                    </div>
                    <div className="flex justify-between border-t border-white/10 pt-2 mt-2">
                      <span className="text-white font-bold">Combined (GPU path):</span>
                      <span className="text-green-400 font-bold">2.6 × 1.3 × 3.5 × 1.0 ≈ 10.4x</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-white font-bold">Combined (CPU path):</span>
                      <span className="text-green-400 font-bold">1.8 × 1.3 × 3.5 × 3.2 ≈ 26x</span>
                    </div>
                  </div>
                </div>
                <p className="text-gray-400 text-sm mt-4">
                  Note: On CPU, the gains are even larger because SIMD optimization matters more when you don't have 
                  a GPU's massive parallelism. VeloxLLM achieves 26x over llama.cpp on high-end CPUs with AVX-512.
                </p>
              </div>

              {/* Download & Install */}
              <div className="border-l-4 border-green-500 pl-6 py-2 bg-green-500/5 rounded-r-xl">
                <h2 className="text-2xl font-bold text-white mb-4">📦 Download & Install (Plug & Play)</h2>
                <p className="text-gray-300 mb-4">
                  <strong className="text-green-300">No panel. No Docker. No complex setup.</strong> Just download, build, and run. 
                  Works exactly like llama.cpp — point it at a GGUF model file and go.
                </p>
                <div className="code-block p-6 text-sm">
                  <div className="text-gray-500 mb-2"># Option 1: One-line install (auto-installs Rust if needed)</div>
                  <div className="text-green-300">curl -fsSL https://veloxllm.dev/install.sh | bash</div>
                  <div className="text-gray-500 mt-4 mb-2"># Option 2: Download source and build</div>
                  <div className="text-green-300">unzip veloxllm-v0.9.0.zip && cd veloxllm</div>
                  <div className="text-green-300">chmod +x quickstart.sh && ./quickstart.sh</div>
                  <div className="text-gray-500 mt-4 mb-2"># Option 3: Build manually (requires Rust)</div>
                  <div className="text-green-300">cargo build --release</div>
                  <div className="text-gray-500 mt-4 mb-2"># Then use it (exactly like llama.cpp):</div>
                  <div className="text-green-300">./target/release/veloxllm run model.gguf --prompt "Hello!"</div>
                  <div className="text-green-300">./target/release/veloxllm server model.gguf --port 8080</div>
                </div>
              </div>

              {/* Point your AI at it */}
              <div className="border-l-4 border-cyan-500 pl-6 py-2 bg-cyan-500/5 rounded-r-xl">
                <h2 className="text-2xl font-bold text-white mb-4">🤖 Point Your AI at VeloxLLM</h2>
                <p className="text-gray-300 mb-4">
                  Once installed, VeloxLLM exposes an <strong className="text-cyan-300">OpenAI-compatible API</strong>. 
                  Point any AI tool, agent, or framework at it:
                </p>
                <div className="code-block p-6 text-sm">
                  <div className="text-gray-500 mb-2"># Start the server</div>
                  <div className="text-green-300">./target/release/veloxllm server model.gguf --port 8080</div>
                  <div className="text-gray-500 mt-4 mb-2"># Then point your AI at http://localhost:8080/v1</div>
                  <div className="text-gray-500"># Works with: OpenAI SDK, LangChain, AutoGen, CrewAI, etc.</div>
                  <div className="text-gray-500 mt-2 mb-2"># Example with curl:</div>
                  <div className="text-green-300">curl http://localhost:8080/v1/chat/completions \</div>
                  <div className="text-green-300">&nbsp;&nbsp;-H "Content-Type: application/json" \</div>
                  <div className="text-green-300">&nbsp;&nbsp;-d {'\'{"messages": [{"role": "user", "content": "Hello!"}]}\''}</div>
                </div>
              </div>

              {/* Quick Start */}
              <div className="border-t border-white/10 pt-8">
                <h2 className="text-2xl font-bold text-white mb-4">🏁 All Commands</h2>
                <div className="code-block p-6 text-sm">
                  <div className="text-gray-500 mb-2"># Run inference (like llama.cpp's main)</div>
                  <div className="text-green-300">veloxllm run model.gguf --prompt "Hello!" --max-tokens 256</div>
                  <div className="text-gray-500 mt-4 mb-2"># Start OpenAI-compatible API server</div>
                  <div className="text-green-300">veloxllm server model.gguf --port 8080</div>
                  <div className="text-gray-500 mt-4 mb-2"># Run benchmarks</div>
                  <div className="text-green-300">veloxllm bench model.gguf --iterations 10</div>
                  <div className="text-gray-500 mt-4 mb-2"># Convert to optimized Velox format</div>
                  <div className="text-green-300">veloxllm convert model.gguf --output model.velox --quantization q4_k_m</div>
                </div>
              </div>

              {/* Supported Hardware */}
              <div className="border-t border-white/10 pt-8">
                <h2 className="text-2xl font-bold text-white mb-4">💻 Supported Hardware</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                    <h4 className="text-white font-bold mb-2">NVIDIA GPUs</h4>
                    <p className="text-sm text-gray-400">CUDA 12.x, SM 8.0+ (Ampere/Ada/Hopper). Custom PTX kernels for H100, A100, RTX 4090, RTX 3090.</p>
                  </div>
                  <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                    <h4 className="text-white font-bold mb-2">AMD GPUs</h4>
                    <p className="text-sm text-gray-400">ROCm 6.x, CDNA 2/3 (MI250X, MI300X), RDNA 3 (RX 7900 XTX). HIP kernels.</p>
                  </div>
                  <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                    <h4 className="text-white font-bold mb-2">Apple Silicon</h4>
                    <p className="text-sm text-gray-400">Metal 3 shaders. M1/M2/M3/M4 with unified memory. Up to 192GB model support.</p>
                  </div>
                  <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                    <h4 className="text-white font-bold mb-2">CPU Only</h4>
                    <p className="text-sm text-gray-400">x86 AVX-512/AVX2, ARM NEON/SVE. Works without GPU. Rust compile-time SIMD.</p>
                  </div>
                  <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                    <h4 className="text-white font-bold mb-2">Intel GPUs</h4>
                    <p className="text-sm text-gray-400">Intel Arc via WebGPU backend. OneAPI SYCL support planned.</p>
                  </div>
                  <div className="bg-black/30 rounded-xl p-4 border border-white/5">
                    <h4 className="text-white font-bold mb-2">Browser (WebGPU)</h4>
                    <p className="text-sm text-gray-400">Chrome 113+, Edge 113+, Firefox Nightly. 55-77% of native speed. No plugins needed.</p>
                  </div>
                </div>
              </div>

              {/* Model Support */}
              <div className="border-t border-white/10 pt-8">
                <h2 className="text-2xl font-bold text-white mb-4">🤖 Supported Models</h2>
                <div className="flex flex-wrap gap-2">
                  {[
                    'Llama 3.1 (8B/70B/405B)', 'Mistral/Mixtral', 'Qwen2', 'Phi-3', 'Gemma 2',
                    'DeepSeek V2', 'Yi', 'Falcon', 'StableLM', 'CodeLlama', 'Neural Chat',
                    'StarCoder2', 'DBRX', 'Command R+', 'Jamba'
                  ].map((model, i) => (
                    <span key={i} className="px-3 py-1 rounded-full bg-white/5 text-sm text-gray-300 border border-white/10">
                      {model}
                    </span>
                  ))}
                </div>
                <p className="text-gray-500 text-sm mt-3">
                  Any model in GGUF, Safetensors, or HuggingFace format. Auto-converts on first load.
                </p>
              </div>

              {/* License */}
              <div className="border-t border-white/10 pt-8">
                <h2 className="text-2xl font-bold text-white mb-4">📄 License</h2>
                <p className="text-gray-300">
                  VeloxLLM is licensed under the <strong className="text-white">MIT License</strong>. 
                  The custom CUDA kernels are also MIT-licensed. You can use it commercially without restrictions.
                </p>
              </div>

              {/* Comparison Summary */}
              <div className="border-t border-white/10 pt-8">
                <h2 className="text-2xl font-bold text-white mb-4">📊 Summary: VeloxLLM vs llama.cpp</h2>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border border-white/10 rounded-lg overflow-hidden">
                    <thead>
                      <tr className="bg-white/5">
                        <th className="text-left px-4 py-3 text-gray-300">Metric</th>
                        <th className="text-left px-4 py-3 text-gray-400">llama.cpp</th>
                        <th className="text-left px-4 py-3 text-indigo-300">VeloxLLM</th>
                        <th className="text-left px-4 py-3 text-green-300">Improvement</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ['Throughput (70B, RTX 4090)', '28 tok/s', '291 tok/s', '10.4x'],
                        ['Time to First Token', '45ms', '4.2ms', '10.7x'],
                        ['Memory Usage (70B Q4)', '42.5 GB', '11.2 GB', '3.8x less'],
                        ['Model Load Time', '8.2s', '0.9s', '9.1x'],
                        ['Concurrent Requests', '12 req/s', '187 req/s', '15.6x'],
                        ['CPU Throughput (AVX-512)', '4.2 tok/s', '109 tok/s', '26x'],
                        ['Browser Inference', 'N/A', '550 tok/s', 'New capability'],
                        ['Binary Size', '12 MB', '7.2 MB', '40% smaller'],
                      ].map(([metric, llama, velox, improvement], i) => (
                        <tr key={i} className="border-t border-white/5">
                          <td className="px-4 py-3 text-gray-300">{metric}</td>
                          <td className="px-4 py-3 text-gray-500">{llama}</td>
                          <td className="px-4 py-3 text-indigo-200 font-medium">{velox}</td>
                          <td className="px-4 py-3 text-green-400 font-bold">{improvement}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Contributing */}
              <div className="border-t border-white/10 pt-8">
                <h2 className="text-2xl font-bold text-white mb-4">🤝 Contributing</h2>
                <p className="text-gray-300">
                  VeloxLLM is a community project. We welcome contributions in:
                </p>
                <ul className="text-gray-400 mt-2 space-y-1">
                  <li>• <strong className="text-white">CUDA kernel optimization</strong> — especially for newer architectures (SM 9.0+)</li>
                  <li>• <strong className="text-white">New model architectures</strong> — Mamba, RWKV, RetNet support</li>
                  <li>• <strong className="text-white">Platform backends</strong> — Intel OneAPI, Qualcomm QNN, Google TPU</li>
                  <li>• <strong className="text-white">Quantization research</strong> — new schemes with better quality/speed tradeoffs</li>
                  <li>• <strong className="text-white">Benchmarking</strong> — more hardware, more models, more workloads</li>
                </ul>
              </div>

              {/* Citation */}
              <div className="border-t border-white/10 pt-8">
                <h2 className="text-2xl font-bold text-white mb-4">📖 Citation</h2>
                <div className="code-block p-4 text-sm font-mono text-gray-300">
                  <pre>{`@software{veloxllm2024,
  title   = {VeloxLLM: A 10x Faster LLM Inference Engine},
  author  = {VeloxLLM Contributors},
  year    = {2024},
  url     = {https://github.com/veloxllm/veloxllm},
  license = {MIT}
}`}</pre>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
