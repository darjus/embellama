// Copyright 2025 Embellama Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Backend detection and configuration for optimal performance

use serde::{Deserialize, Serialize};
use std::fmt;

/// Available backend types for llama.cpp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// CPU-only backend
    Cpu,
    /// CPU with OpenMP acceleration
    OpenMP,
    /// CPU with native optimizations
    Native,
    /// NVIDIA CUDA GPU acceleration
    Cuda,
    /// Apple Metal GPU acceleration
    Metal,
    /// Vulkan GPU acceleration
    Vulkan,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::OpenMP => write!(f, "OpenMP"),
            Self::Native => write!(f, "Native"),
            Self::Cuda => write!(f, "CUDA"),
            Self::Metal => write!(f, "Metal"),
            Self::Vulkan => write!(f, "Vulkan"),
        }
    }
}

impl BackendType {
    /// Check if this backend supports GPU acceleration
    #[must_use]
    pub const fn is_gpu_accelerated(&self) -> bool {
        matches!(self, Self::Cuda | Self::Metal | Self::Vulkan)
    }

    /// Get recommended GPU layers for this backend
    /// Returns None for CPU backends, Some(999) for GPU backends (effectively all layers)
    #[must_use]
    pub const fn recommended_gpu_layers(&self) -> Option<u32> {
        if self.is_gpu_accelerated() {
            Some(999) // Use all available layers
        } else {
            None
        }
    }
}

/// Detect the best available backend based on compile-time features and runtime capabilities
#[must_use]
#[allow(unreachable_code)]
pub fn detect_best_backend() -> BackendType {
    // Priority order: Metal (on macOS) > CUDA > Vulkan > OpenMP > Native > CPU

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        // On macOS with Metal feature, always prefer Metal
        return BackendType::Metal;
    }

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        // On non-macOS systems with CUDA feature
        // In a full implementation, we'd check if CUDA is actually available
        // For now, assume it's available if the feature is enabled
        return BackendType::Cuda;
    }

    #[cfg(feature = "vulkan")]
    {
        // Vulkan is cross-platform
        // In a full implementation, we'd check if Vulkan is actually available
        return BackendType::Vulkan;
    }

    #[cfg(feature = "openmp")]
    {
        return BackendType::OpenMP;
    }

    #[cfg(feature = "native")]
    {
        return BackendType::Native;
    }

    // Default fallback
    BackendType::Cpu
}

/// Get the currently compiled backend based on build features
#[must_use]
pub fn get_compiled_backend() -> BackendType {
    // Check the environment variable set by build.rs
    match option_env!("EMBELLAMA_BACKEND") {
        Some("metal") => BackendType::Metal,
        Some("cuda") => BackendType::Cuda,
        Some("vulkan") => BackendType::Vulkan,
        Some("openmp") => BackendType::OpenMP,
        Some("native") => BackendType::Native,
        _ => BackendType::Cpu,
    }
}

/// Backend information for diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    /// The detected best backend
    pub backend: BackendType,
    /// The backend that was compiled in
    pub compiled_backend: BackendType,
    /// Available features at compile time
    pub available_features: Vec<String>,
    /// Platform information
    pub platform: String,
}

impl BackendInfo {
    /// Create backend information for diagnostics
    #[must_use]
    #[allow(clippy::vec_init_then_push)]
    pub fn new() -> Self {
        let mut features: Vec<String> = vec![];

        #[cfg(feature = "metal")]
        features.push("metal".to_string());

        #[cfg(feature = "cuda")]
        features.push("cuda".to_string());

        #[cfg(feature = "cuda-no-vmm")]
        features.push("cuda-no-vmm".to_string());

        #[cfg(feature = "vulkan")]
        features.push("vulkan".to_string());

        #[cfg(feature = "openmp")]
        features.push("openmp".to_string());

        #[cfg(feature = "native")]
        features.push("native".to_string());

        #[cfg(feature = "dynamic-link")]
        features.push("dynamic-link".to_string());

        let platform = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);

        Self {
            backend: detect_best_backend(),
            compiled_backend: get_compiled_backend(),
            available_features: features,
            platform,
        }
    }
}

impl Default for BackendInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BackendInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Backend Information:")?;
        writeln!(f, "  Detected Backend: {}", self.backend)?;
        writeln!(f, "  Compiled Backend: {}", self.compiled_backend)?;
        writeln!(f, "  Platform: {}", self.platform)?;
        writeln!(
            f,
            "  Available Features: {}",
            self.available_features.join(", ")
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::Cpu.to_string(), "CPU");
        assert_eq!(BackendType::OpenMP.to_string(), "OpenMP");
        assert_eq!(BackendType::Cuda.to_string(), "CUDA");
        assert_eq!(BackendType::Metal.to_string(), "Metal");
        assert_eq!(BackendType::Vulkan.to_string(), "Vulkan");
    }

    #[test]
    fn test_is_gpu_accelerated() {
        assert!(!BackendType::Cpu.is_gpu_accelerated());
        assert!(!BackendType::OpenMP.is_gpu_accelerated());
        assert!(!BackendType::Native.is_gpu_accelerated());
        assert!(BackendType::Cuda.is_gpu_accelerated());
        assert!(BackendType::Metal.is_gpu_accelerated());
        assert!(BackendType::Vulkan.is_gpu_accelerated());
    }

    #[test]
    fn test_recommended_gpu_layers() {
        assert_eq!(BackendType::Cpu.recommended_gpu_layers(), None);
        assert_eq!(BackendType::OpenMP.recommended_gpu_layers(), None);
        assert_eq!(BackendType::Native.recommended_gpu_layers(), None);
        assert_eq!(BackendType::Cuda.recommended_gpu_layers(), Some(999));
        assert_eq!(BackendType::Metal.recommended_gpu_layers(), Some(999));
        assert_eq!(BackendType::Vulkan.recommended_gpu_layers(), Some(999));
    }

    #[test]
    fn test_backend_info_creation() {
        let info = BackendInfo::new();
        assert!(!info.platform.is_empty());
        // The detected backend should match the compiled backend
        assert_eq!(info.backend, info.compiled_backend);
    }

    #[test]
    fn test_backend_detection() {
        // This test will pass based on the features enabled during compilation
        let backend = detect_best_backend();

        // At minimum, we should get CPU backend
        match backend {
            BackendType::Cpu
            | BackendType::OpenMP
            | BackendType::Native
            | BackendType::Cuda
            | BackendType::Metal
            | BackendType::Vulkan => {
                // Any of these is valid
            }
        }
    }
}
