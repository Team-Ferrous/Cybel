// highnoon/_native/ops/quantum_curriculum_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
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

/**
 * @file quantum_curriculum_op.h
 * @brief Phase 200+: SAQC (Spectrally-Aware Quantum Curriculum) C++ ops.
 *
 * HIGHNOON_UPGRADE_ROADMAP.md complementary phase.
 * Synergizes with QULS telemetry to drive curriculum decisions.
 *
 * Provides FFT-domain spectral complexity analysis for O(D log D) curriculum
 * scoring. Used by QuantumSynergyCurriculum to determine data progression
 * based on model quantum state.
 */

#ifndef HIGHNOON_NATIVE_OPS_QUANTUM_CURRICULUM_OP_H_
#define HIGHNOON_NATIVE_OPS_QUANTUM_CURRICULUM_OP_H_

#include <cmath>
#include <complex>
#include <vector>

namespace hsmn {
namespace saqc {

// =============================================================================
// Configuration Structure
// =============================================================================

/**
 * @brief Configuration for SAQC spectral analysis.
 */
struct SAQCConfig {
    int fft_dim = 64;                      // FFT dimension for spectral analysis
    float entropy_threshold = 0.5f;        // Normalized entropy threshold
    float fidelity_threshold = 0.85f;      // Fidelity threshold for acceleration
    float coherence_threshold = 0.9f;      // Coherence threshold for gating
    int hidden_dim = 512;                  // Model hidden dimension
    bool use_power_spectrum = true;        // Use power spectrum (vs raw FFT)
    int num_eigenvalues = 8;               // Top eigenvalues for complexity
};

// =============================================================================
// Curriculum Mode Enumeration
// =============================================================================

/**
 * @brief SAQC operating modes driven by quantum telemetry.
 */
enum class CurriculumMode {
    NORMAL = 0,      // Standard adaptive progression
    RETREAT = 1,     // Low entropy → diversity restoration
    TUNNELING = 2,   // Barren plateau → orthogonal injection
    ACCELERATE = 3   // High fidelity → rapid advancement
};

// =============================================================================
// Core Kernel Declarations
// =============================================================================

/**
 * @brief Compute spectral complexity scores via FFT-domain analysis.
 *
 * Analyzes hidden state representations to determine spectral complexity
 * for curriculum difficulty assignment. Uses power spectrum analysis.
 *
 * Complexity: O(batch × D log D) via FFT
 *
 * @param representations Input representations [batch, hidden_dim]
 * @param complexity_scores Output complexity scores [batch]
 * @param spectral_entropy Output normalized entropy [batch]
 * @param config SAQC configuration
 * @param batch_size Number of samples
 */
void ComputeSpectralComplexity(
    const float* representations,
    float* complexity_scores,
    float* spectral_entropy,
    const SAQCConfig& config,
    int batch_size
);

/**
 * @brief Compute gradient for spectral complexity (backprop).
 *
 * @param grad_complexity Gradient w.r.t. complexity scores [batch]
 * @param representations Original input [batch, hidden_dim]
 * @param grad_representations Output gradient [batch, hidden_dim]
 * @param config SAQC configuration
 * @param batch_size Number of samples
 */
void ComputeSpectralComplexityGrad(
    const float* grad_complexity,
    const float* representations,
    float* grad_representations,
    const SAQCConfig& config,
    int batch_size
);

/**
 * @brief Determine curriculum mode from QULS telemetry metrics.
 *
 * Implements SAQC decision logic:
 *   - barren_plateau → TUNNELING
 *   - low entropy → RETREAT
 *   - high fidelity + high coherence → ACCELERATE
 *   - otherwise → NORMAL
 *
 * @param spectral_entropy Normalized spectral entropy
 * @param fidelity_loss Fidelity loss from QULS
 * @param coherence Coherence metric from QULS
 * @param barren_plateau_detected Barren plateau flag
 * @param config SAQC configuration
 * @return CurriculumMode Determined operating mode
 */
CurriculumMode DetermineCurriculumMode(
    float spectral_entropy,
    float fidelity_loss,
    float coherence,
    bool barren_plateau_detected,
    const SAQCConfig& config
);

// =============================================================================
// FFT Helper Functions
// =============================================================================

/**
 * @brief Compute power spectrum via real-valued FFT.
 *
 * @param input Real input signal [dim]
 * @param power_spectrum Output power spectrum [dim/2 + 1]
 * @param dim Signal dimension
 */
void ComputePowerSpectrum(
    const float* input,
    float* power_spectrum,
    int dim
);

/**
 * @brief Compute normalized spectral entropy from power spectrum.
 *
 * H = -Σ p_i log(p_i) / log(N), where p_i = |X_i|² / Σ|X_j|²
 *
 * @param power_spectrum Power spectrum values
 * @param spectrum_size Size of spectrum
 * @return Normalized entropy in [0, 1]
 */
float ComputeNormalizedEntropy(
    const float* power_spectrum,
    int spectrum_size
);

}  // namespace saqc
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_QUANTUM_CURRICULUM_OP_H_
