// VeloxLLM Model Loader
// Supports GGUF, Safetensors, and native Velox format.
// Uses memory-mapped files for zero-copy loading.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use tracing::info;

/// Model configuration extracted from the model file
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub total_params: u64,
    pub memory_bytes: u64,
    pub quantization: Quantization,
}

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    F16,
    Q8_0,
    Q6_K,
    Q5_K_M,
    Q4_K_M,
    Q4_0,
    Q3_K_M,
    Q2_K,
}

/// The loaded model with memory-mapped weights
pub struct Model {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    _mmap: Option<Mmap>, // Keep mmap alive
}

/// Model weights - memory-mapped for zero-copy access
pub struct ModelWeights {
    pub embedding: WeightRef,
    pub layers: Vec<TransformerLayer>,
    pub output_norm: WeightRef,
    pub output: WeightRef,
}

pub struct WeightRef {
    pub data: *const u8,
    pub len: usize,
}

// SAFETY: WeightRef is read-only and the data is backed by a memory-mapped file
unsafe impl Send for WeightRef {}
unsafe impl Sync for WeightRef {}

pub struct TransformerLayer {
    pub attention_norm: WeightRef,
    pub wq: WeightRef,
    pub wk: WeightRef,
    pub wv: WeightRef,
    pub wo: WeightRef,
    pub ffn_norm: WeightRef,
    pub w_gate: WeightRef,
    pub w_up: WeightRef,
    pub w_down: WeightRef,
}

impl Model {
    /// Load a model from disk
    pub async fn load(path: &Path, gpu_layers: i32, ctx_size: usize) -> Result<Self> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension {
            "gguf" => Self::load_gguf(path, gpu_layers, ctx_size).await,
            "safetensors" | "bin" => Self::load_safetensors(path, gpu_layers, ctx_size).await,
            "velox" => Self::load_velox(path, gpu_layers, ctx_size).await,
            _ => bail!("Unsupported model format: .{}", extension),
        }
    }

    /// Load from GGUF format (compatible with llama.cpp models)
    async fn load_gguf(path: &Path, gpu_layers: i32, ctx_size: usize) -> Result<Self> {
        info!("Loading GGUF model: {}", path.display());

        let file = File::open(path)
            .context("Failed to open model file")?;

        let mmap = unsafe { Mmap::map(&file) }
            .context("Failed to memory-map model file")?;

        // Parse GGUF header
        let config = crate::gguf::parse_gguf_header(&mmap)?;

        let weights = crate::gguf::parse_gguf_weights(&mmap, &config)?;

        info!("Loaded {} layers, {} params",
            config.num_layers, config.total_params);

        Ok(Self {
            config,
            weights,
            _mmap: Some(mmap),
        })
    }

    /// Load from Safetensors format
    async fn load_safetensors(path: &Path, gpu_layers: i32, ctx_size: usize) -> Result<Self> {
        info!("Loading Safetensors model: {}", path.display());

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        // Parse safetensors header (JSON)
        let config = crate::gguf::parse_safetensors_header(&mmap)?;
        let weights = crate::gguf::parse_safetensors_weights(&mmap, &config)?;

        Ok(Self {
            config,
            weights,
            _mmap: Some(mmap),
        })
    }

    /// Load from native Velox format (optimized, fused weights)
    async fn load_velox(path: &Path, gpu_layers: i32, ctx_size: usize) -> Result<Self> {
        info!("Loading Velox model: {}", path.display());

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;

        // Velox format has pre-fused weights for maximum performance
        let config = crate::gguf::parse_velox_header(&mmap)?;
        let weights = crate::gguf::parse_velox_weights(&mmap, &config)?;

        Ok(Self {
            config,
            weights,
            _mmap: Some(mmap),
        })
    }
}

/// Convert a model to Velox format
pub async fn convert_model(input: &Path, output: &Path, quantization: &str) -> Result<()> {
    use std::io::Write;

    info!("Converting {} -> {}", input.display(), output.display());

    // Load the source model
    let model = Model::load(input, -1, 4096).await?;

    // Write Velox format
    let mut file = File::create(output)?;

    // Write header
    let header = crate::gguf::create_velox_header(&model.config, quantization);
    file.write_all(&header)?;

    // Write fused weights
    let weights = crate::gguf::fuse_weights(&model.weights, quantization)?;
    file.write_all(&weights)?;

    info!("Conversion complete: {:.1} MB", output.metadata()?.len() as f64 / 1_000_000.0);

    Ok(())
}
