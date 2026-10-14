import { useState } from 'react';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

export function DownloadSection() {
  const [downloading, setDownloading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [downloaded, setDownloaded] = useState(false);

  const handleDownload = async () => {
    setDownloading(true);
    setProgress(0);
    setDownloaded(false);

    try {
      const zip = new JSZip();
      const folder = zip.folder('veloxllm')!;

      // Fetch all source files from the public directory
      const filesToFetch = [
        'Cargo.toml',
        'README.md',
        'Makefile',
        'install.sh',
        'src/main.rs',
        'src/arena.rs',
        'src/inference.rs',
        'src/model.rs',
        'src/gguf.rs',
        'src/tokenizer.rs',
        'src/sampler.rs',
        'src/kernels.rs',
        'src/server.rs',
      ];

      const total = filesToFetch.length;

      for (let i = 0; i < filesToFetch.length; i++) {
        const filePath = filesToFetch[i];
        try {
          const response = await fetch(`/veloxllm/${filePath}`);
          if (response.ok) {
            const content = await response.text();
            folder.file(filePath, content);
          }
        } catch {
          // File might not be available, skip
        }
        setProgress(Math.round(((i + 1) / total) * 80));
      }

      // Add quickstart script
      folder.file('quickstart.sh', `#!/bin/bash
# VeloxLLM Quick Start - Build and run in one command
set -e

echo "⚡ VeloxLLM Quick Start"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo "📦 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "\$HOME/.cargo/env"
fi

echo "🔧 Building (optimized, ~2 min)..."
cargo build --release

echo ""
echo "✅ Done! Binary at: ./target/release/veloxllm"
echo ""
echo "Usage (exactly like llama.cpp):"
echo "  ./target/release/veloxllm run your_model.gguf --prompt \\"Hello!\\""
echo "  ./target/release/veloxllm server your_model.gguf --port 8080"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
`);

      // Add Windows batch file
      folder.file('quickstart.bat', `@echo off
echo ⚡ VeloxLLM Quick Start (Windows)
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

where cargo >nul 2>nul
if %errorlevel% neq 0 (
    echo 📦 Please install Rust first: https://rustup.rs
    pause
    exit /b 1
)

echo 🔧 Building...
cargo build --release

echo.
echo ✅ Done! Binary at: target\\release\\veloxllm.exe
echo.
echo Usage:
echo   target\\release\\veloxllm.exe run model.gguf --prompt "Hello!"
echo   target\\release\\veloxllm.exe server model.gguf --port 8080
echo.
pause
`);

      // Add .gitignore
      folder.file('.gitignore', `/target
Cargo.lock
*.velox
`);

      setProgress(90);

      // Generate zip
      const blob = await zip.generateAsync({ 
        type: 'blob',
        compression: 'DEFLATE',
        compressionOptions: { level: 9 }
      }, (metadata) => {
        setProgress(90 + Math.round(metadata.percent / 10));
      });

      setProgress(100);
      saveAs(blob, 'veloxllm-v0.9.0.zip');
      setDownloaded(true);

      setTimeout(() => {
        setDownloading(false);
        setProgress(0);
      }, 2000);

    } catch (error) {
      console.error('Download failed:', error);
      setDownloading(false);
      setProgress(0);
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      {/* Main Download Card */}
      <div className="glass-card rounded-2xl p-8 glow-border mb-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/20 text-green-300 text-xs font-medium border border-green-500/30 mb-4">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
              Ready to download
            </div>
            <h2 className="text-3xl font-bold text-white mb-3">
              Download & Run in 30 Seconds
            </h2>
            <p className="text-gray-400 leading-relaxed mb-4">
              Complete Rust project with all source code. Build once, run forever. 
              Works <strong className="text-white">exactly like llama.cpp</strong> — same models, same workflow, just faster.
            </p>
            <div className="flex flex-wrap gap-2 mb-6">
              <span className="px-2 py-1 rounded bg-white/5 text-xs text-gray-400 border border-white/10">📁 ~50KB source</span>
              <span className="px-2 py-1 rounded bg-white/5 text-xs text-gray-400 border border-white/10">🦀 Rust</span>
              <span className="px-2 py-1 rounded bg-white/5 text-xs text-gray-400 border border-white/10">⚡ CUDA/Metal/CPU</span>
              <span className="px-2 py-1 rounded bg-white/5 text-xs text-gray-400 border border-white/10">📄 MIT</span>
              <span className="px-2 py-1 rounded bg-white/5 text-xs text-gray-400 border border-white/10">🔌 Plug & Play</span>
            </div>
          </div>

          <div className="text-center">
            <button
              onClick={handleDownload}
              disabled={downloading}
              className={`
                relative w-full px-8 py-5 rounded-xl font-bold text-lg transition-all
                ${downloading 
                  ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
                  : downloaded
                    ? 'bg-green-600 text-white'
                    : 'bg-gradient-to-r from-indigo-500 to-cyan-500 text-white hover:opacity-90 hover:scale-[1.02] shadow-lg shadow-indigo-500/25 cursor-pointer'
                }
              `}
            >
              {downloading ? (
                <span className="flex items-center justify-center gap-3">
                  <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Packaging... {progress}%
                </span>
              ) : downloaded ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Downloaded! Check your downloads folder
                </span>
              ) : (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download veloxllm-v0.9.0.zip
                </span>
              )}
            </button>

            {downloading && (
              <div className="mt-4 w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-indigo-500 to-cyan-500 transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* After Download Instructions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-3 mb-3">
            <span className="w-8 h-8 rounded-full bg-indigo-500/20 text-indigo-300 flex items-center justify-center font-bold text-sm">1</span>
            <h4 className="text-white font-bold">Extract & Build</h4>
          </div>
          <div className="code-block p-3 text-xs font-mono text-green-300">
            <div>unzip veloxllm-v0.9.0.zip</div>
            <div>cd veloxllm</div>
            <div>chmod +x quickstart.sh</div>
            <div>./quickstart.sh</div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Or just: <code className="text-gray-400">cargo build --release</code>
          </p>
        </div>

        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-3 mb-3">
            <span className="w-8 h-8 rounded-full bg-cyan-500/20 text-cyan-300 flex items-center justify-center font-bold text-sm">2</span>
            <h4 className="text-white font-bold">Run (like llama.cpp)</h4>
          </div>
          <div className="code-block p-3 text-xs font-mono text-green-300">
            <div className="text-gray-500"># Run inference:</div>
            <div>./target/release/veloxllm \</div>
            <div>&nbsp;&nbsp;run model.gguf \</div>
            <div>&nbsp;&nbsp;--prompt "Hello!"</div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Same GGUF models as llama.cpp
          </p>
        </div>

        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center gap-3 mb-3">
            <span className="w-8 h-8 rounded-full bg-green-500/20 text-green-300 flex items-center justify-center font-bold text-sm">3</span>
            <h4 className="text-white font-bold">Or Start Server</h4>
          </div>
          <div className="code-block p-3 text-xs font-mono text-green-300">
            <div className="text-gray-500"># OpenAI-compatible API:</div>
            <div>./target/release/veloxllm \</div>
            <div>&nbsp;&nbsp;server model.gguf \</div>
            <div>&nbsp;&nbsp;--port 8080</div>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Works with any OpenAI client
          </p>
        </div>
      </div>

      {/* What's in the zip */}
      <div className="mt-8 glass-card rounded-xl p-6">
        <h4 className="text-white font-bold mb-4">📦 What's in the download:</h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
          {[
            { file: 'src/main.rs', desc: 'CLI entry point (run/server/bench/convert)' },
            { file: 'src/inference.rs', desc: 'Core inference engine with speculative decoding' },
            { file: 'src/kernels.rs', desc: 'Fused compute kernels (replaces 7 llama.cpp kernels)' },
            { file: 'src/arena.rs', desc: 'Zero-allocation memory arena' },
            { file: 'src/model.rs', desc: 'Model loader (GGUF/Safetensors/Velox)' },
            { file: 'src/tokenizer.rs', desc: 'BPE tokenizer' },
            { file: 'src/sampler.rs', desc: 'Temperature/top-k/top-p sampling' },
            { file: 'src/server.rs', desc: 'OpenAI-compatible HTTP server' },
            { file: 'src/gguf.rs', desc: 'GGUF format parser' },
            { file: 'Cargo.toml', desc: 'Rust dependencies & build config' },
            { file: 'Makefile', desc: 'Simple build commands' },
            { file: 'quickstart.sh', desc: 'One-command build script' },
          ].map((item, i) => (
            <div key={i} className="flex items-start gap-2 py-1">
              <code className="text-indigo-300 text-xs font-mono whitespace-nowrap">{item.file}</code>
              <span className="text-gray-500 text-xs">— {item.desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* No dependencies note */}
      <div className="mt-6 text-center">
        <p className="text-gray-500 text-sm">
          ✅ No panel to install • No Docker • No complex setup • Just build and run
        </p>
        <p className="text-gray-600 text-xs mt-1">
          Requires only Rust (auto-installed by quickstart.sh if missing)
        </p>
      </div>
    </div>
  );
}
