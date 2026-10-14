// VeloxLLM - The fastest open-source LLM inference engine
// Plug-and-play like llama.cpp, but 10x faster.
//
// Usage:
//   veloxllm run <model.gguf> --prompt "Hello, world!"
//   veloxllm server <model.gguf> --port 8080
//   veloxllm bench <model.gguf>

mod arena;
mod gguf;
mod inference;
mod kernels;
mod model;
mod sampler;
mod server;
mod tokenizer;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "veloxllm")]
#[command(about = "⚡ VeloxLLM - 10x faster LLM inference engine")]
#[command(version = "0.9.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model (like llama.cpp's main)
    Run {
        /// Path to the model file (GGUF, Safetensors, or Velox format)
        model: PathBuf,

        /// Prompt text
        #[arg(short, long, default_value = "Hello!")]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short = 'n', long, default_value = "256")]
        max_tokens: usize,

        /// Temperature (0 = greedy)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Top-K sampling
        #[arg(long, default_value = "40")]
        top_k: usize,

        /// Top-P (nucleus) sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Number of GPU layers to offload (-1 = all)
        #[arg(short = 'g', long, default_value = "-1")]
        gpu_layers: i32,

        /// Context window size
        #[arg(short = 'c', long, default_value = "4096")]
        ctx_size: usize,

        /// Number of threads (0 = auto)
        #[arg(short = 't', long, default_value = "0")]
        threads: usize,

        /// Enable speculative decoding
        #[arg(long, default_value = "true")]
        speculative: bool,

        /// Draft model path for speculative decoding
        #[arg(long)]
        draft_model: Option<PathBuf>,
    },

    /// Start an OpenAI-compatible API server
    Server {
        /// Path to the model file
        model: PathBuf,

        /// Port to listen on
        #[arg(short, long, default_value = "8080")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Number of GPU layers to offload
        #[arg(short = 'g', long, default_value = "-1")]
        gpu_layers: i32,

        /// Context window size
        #[arg(short = 'c', long, default_value = "4096")]
        ctx_size: usize,

        /// Max concurrent requests
        #[arg(long, default_value = "32")]
        max_concurrent: usize,
    },

    /// Run benchmarks on a model
    Bench {
        /// Path to the model file
        model: PathBuf,

        /// Number of benchmark iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,

        /// Prompt length for benchmark
        #[arg(long, default_value = "512")]
        prompt_len: usize,
    },

    /// Convert a model from GGUF/Safetensors to Velox format (optimized)
    Convert {
        /// Input model path
        input: PathBuf,

        /// Output path
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization level (q2_k, q3_k, q4_k_m, q5_k_m, q6_k, q8_0, f16)
        #[arg(short, long, default_value = "q4_k_m")]
        quantization: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive("veloxllm=info".parse().unwrap()))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            model, prompt, max_tokens, temperature, top_k, top_p,
            gpu_layers, ctx_size, threads, speculative, draft_model,
        } => {
            println!("⚡ VeloxLLM v0.9.0");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("📦 Loading model: {}", model.display());

            let start = std::time::Instant::now();
            let engine = inference::VeloxEngine::load(
                &model,
                inference::EngineConfig {
                    gpu_layers,
                    ctx_size,
                    threads: if threads == 0 { num_cpus::get() } else { threads },
                    speculative,
                    draft_model_path: draft_model,
                },
            ).await?;

            let load_time = start.elapsed();
            println!("✅ Model loaded in {:.2}s", load_time.as_secs_f64());
            println!("🧠 Architecture: {}", engine.architecture_name());
            println!("📊 Parameters: {}", engine.param_count());
            println!("💾 Memory: {:.1} GB", engine.memory_usage_gb());
            println!("🎯 Device: {}", engine.device_name());
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!();

            // Generate
            print!("{}", &prompt);
            let tokens = engine.generate(
                &prompt,
                max_tokens,
                sampler::SamplerConfig {
                    temperature,
                    top_k,
                    top_p,
                },
            ).await?;

            println!();
            println!();
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("⚡ Generated {} tokens", tokens.len());
        }

        Commands::Server {
            model, port, host, gpu_layers, ctx_size, max_concurrent,
        } => {
            println!("⚡ VeloxLLM Server v0.9.0");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("📦 Loading model: {}", model.display());

            let engine = inference::VeloxEngine::load(
                &model,
                inference::EngineConfig {
                    gpu_layers,
                    ctx_size,
                    threads: num_cpus::get(),
                    speculative: true,
                    draft_model_path: None,
                },
            ).await?;

            println!("✅ Model loaded");
            println!("🌐 Server starting at http://{}:{}", host, port);
            println!("📡 OpenAI-compatible API at /v1/chat/completions");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            server::start_server(engine, &host, port, max_concurrent).await?;
        }

        Commands::Bench { model, iterations, prompt_len } => {
            println!("⚡ VeloxLLM Benchmark v0.9.0");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            let engine = inference::VeloxEngine::load(
                &model,
                inference::EngineConfig {
                    gpu_layers: -1,
                    ctx_size: 4096,
                    threads: num_cpus::get(),
                    speculative: true,
                    draft_model_path: None,
                },
            ).await?;

            engine.run_benchmark(iterations, prompt_len).await?;
        }

        Commands::Convert { input, output, quantization } => {
            println!("⚡ VeloxLLM Converter v0.9.0");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("📦 Converting: {}", input.display());
            println!("📤 Output: {}", output.display());
            println!("🎯 Quantization: {}", quantization);

            model::convert_model(&input, &output, &quantization).await?;

            println!("✅ Conversion complete!");
        }
    }

    Ok(())
}
