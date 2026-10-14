// VeloxLLM Token Sampler
// Implements temperature, top-k, top-p (nucleus), and repetition penalty sampling.

use anyhow::Result;

pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

pub struct Sampler {
    // Repetition penalty state
    recent_tokens: Vec<u32>,
    max_recent: usize,
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            recent_tokens: Vec::new(),
            max_recent: 64,
        }
    }

    /// Sample a token from logits
    pub fn sample(&mut self, logits: &[f32], config: &SamplerConfig) -> Result<u32> {
        let mut probs = self.logits_to_probs(logits, config)?;

        // Apply repetition penalty
        self.apply_repetition_penalty(&mut probs);

        // Sample from distribution
        let token = self.sample_from_probs(&probs)?;

        // Track recent tokens
        self.recent_tokens.push(token);
        if self.recent_tokens.len() > self.max_recent {
            self.recent_tokens.remove(0);
        }

        Ok(token)
    }

    /// Convert logits to probabilities with temperature and filtering
    fn logits_to_probs(&self, logits: &[f32], config: &SamplerConfig) -> Result<Vec<f32>> {
        let vocab_size = logits.len();

        // Apply temperature
        let scaled: Vec<f32> = if config.temperature > 0.0 {
            logits.iter().map(|&l| l / config.temperature).collect()
        } else {
            logits.to_vec()
        };

        // Find max for numerical stability
        let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute softmax
        let exp_sum: f32 = scaled.iter()
            .map(|&l| (l - max_logit).exp())
            .sum();

        let mut probs: Vec<f32> = scaled.iter()
            .map(|&l| (l - max_logit).exp() / exp_sum)
            .collect();

        // Apply top-k filtering
        if config.top_k > 0 && config.top_k < vocab_size {
            let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Zero out everything beyond top-k
            for &(_, p) in &indexed[config.top_k..] {
                // Find and zero out
                for prob in probs.iter_mut() {
                    if (*prob - p).abs() < 1e-10 {
                        *prob = 0.0;
                    }
                }
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        // Apply top-p (nucleus) filtering
        if config.top_p < 1.0 {
            let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cumsum = 0.0;
            let mut cutoff_idx = indexed.len();

            for (i, &(_, p)) in indexed.iter().enumerate() {
                cumsum += p;
                if cumsum >= config.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Zero out everything beyond nucleus
            let keep_set: Vec<usize> = indexed[..cutoff_idx].iter().map(|&(idx, _)| idx).collect();

            for (i, p) in probs.iter_mut().enumerate() {
                if !keep_set.contains(&i) {
                    *p = 0.0;
                }
            }

            // Renormalize
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }
        }

        Ok(probs)
    }

    /// Apply repetition penalty to recent tokens
    fn apply_repetition_penalty(&self, probs: &mut [f32]) {
        let penalty = 1.1; // Standard repetition penalty

        for &token in &self.recent_tokens {
            let idx = token as usize;
            if idx < probs.len() {
                // Penalize: reduce probability of repeated tokens
                probs[idx] /= penalty;
            }
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    /// Sample from a probability distribution
    fn sample_from_probs(&self, probs: &[f32]) -> Result<u32> {
        // Greedy (temperature = 0)
        if probs.iter().all(|&p| p == 0.0 || p == 1.0) {
            return Ok(probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0));
        }

        // Random sampling
        let rand = rand_f32();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand < cumsum {
                return Ok(i as u32);
            }
        }

        // Fallback to argmax
        Ok(probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0))
    }
}

/// Simple random f32 in [0, 1)
fn rand_f32() -> f32 {
    use std::time::SystemTime;
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (seed as f32 / u32::MAX as f32)
}
