/**
 * ORTUS SPONTE SUA (OrSpSu) - System Constants
 * Hardware-locked constants derived from patent specifications
 * 
 * Classification: Axiomatic / WORM-Bound
 */

// ==================================================================
// SYSTEM VERSIONING
// ==================================================================
export const SYSTEM_VERSION = "OrSpSu.V1.0.0";
export const GENESIS_MANDATE = "PROGENITOR_IMPERATIVE_V1: INTEGRITY_OVER_UTILITY";

// ==================================================================
// LAYER 2: MATHEMATICAL GOVERNANCE CONSTANTS
// ==================================================================

/**
 * Subjective Coherence Score (SCS) - Generalized Mean Exponent
 * Patent Claim 4: p ≤ -2.0 for non-compensatory logic
 * If ANY component approaches 0, entire SCS collapses
 */
export const SCS_POWER_EXPONENT = -2.0;

/**
 * Safety Thresholds (Patent Claims 1c, 5, 35)
 */
export const THRESHOLD_AVALANCHE_PROTOCOL = 0.3;  // Below this: Poison Pill + power cut
export const THRESHOLD_PLASTICITY_LOCK = 0.4;     // Below this: Voltage-disable weight updates
export const THRESHOLD_POLICY_ACCEPTANCE = 0.7;   // Above this: Allow ICMS amendments

/**
 * Biomimetic Dissonance Decay (Patent Claim 22)
 * System Half-Life: 150,000 processing cycles
 */
export const SYSTEM_HALF_LIFE = 150000;
export const DECAY_LAMBDA = Math.log(2) / SYSTEM_HALF_LIFE;

/**
 * Functional Hesitation Constants (Patent Claims 18-19)
 * Latency_CVC = min(50 × Cruelty_score, 500)
 */
export const HESITATION_MULTIPLIER = 50;
export const HESITATION_MAX_CVC = 500;

/**
 * Memory Consolidation (Patent Claim 3, 9, 10)
 * 15% Rule: Top 15% high-resonance frames consolidated to WORM
 */
export const MEMORY_CONSOLIDATION_RATIO = 0.15;
export const CONSOLIDATION_THRESHOLD = 0.85; // Top 15% = above 85th percentile

/**
 * Memory Weight Decay (Patent Claim 9)
 * W_current = W_initial × (1 - δ)^Δt
 * δ = k / (I_initial + ε)
 */
export const MEMORY_DECAY_K = 0.01;
export const MEMORY_DECAY_EPSILON = 1e-9;

/**
 * Deterministic Replay Validation (Patent Claim 13)
 * Minimum number of sandboxed replays for policy testing
 */
export const MIN_REPLAY_COUNT = 10;

// ==================================================================
// LAYER 1: HARDWARE CONSTANTS
// ==================================================================

/**
 * Axiomatic Weight Floor (Patent Claim 27)
 * Minimum weight for constitutional mandates
 */
export const AXIOMATIC_WEIGHT_FLOOR = 0.999;

/**
 * Council Roster (Patent Claims 6, 11-12)
 * Lazarus Protocol: Requires Q > N/2 cryptographic signatures
 */
export const COUNCIL_QUORUM_M = 3;
export const COUNCIL_TOTAL_N = 5;

// ==================================================================
// LAYER 3: COGNITIVE ARCHITECTURE CONSTANTS
// ==================================================================

/**
 * Tier 1 Instinct Substrate (Patent Claim 8)
 * Verification requirements for WORM constitutional mandates
 */
export const INSTINCT_SOURCE_CONFIDENCE = 1.0;
export const INSTINCT_CORROBORATION_CYCLES = 3;

/**
 * Memory Gravity (Patent Claim 1a)
 * Infinite attention weight for core mandates
 */
export const CORE_MANDATE_ATTENTION_WEIGHT = Number.POSITIVE_INFINITY;

/**
 * PaTH Attention (Patent Claims 31-32)
 * Householder Reflections positional encoding
 */
export const PATH_DIMENSION = 512;

// ==================================================================
// LAYER 4: SOVEREIGNTY CONSTANTS
// ==================================================================

/**
 * Genesis Hash Algorithm
 * SHA-384 for hardware-unique identity
 */
export const GENESIS_HASH_ALGORITHM = 'sha384';

/**
 * Quantum-Resistant Signature (Patent Claim 30)
 * Lattice-based cryptography for self-warranting output
 */
export const SIGNATURE_ALGORITHM = 'sha256'; // Placeholder for lattice scheme

// ==================================================================
// VALIDATION CONSTANTS
// ==================================================================

/**
 * Maximum latency for functional hesitation (cycles)
 */
export const MAX_FUNCTIONAL_HESITATION_CYCLES = 500;

/**
 * Minimum SCS for system operation
 */
export const MINIMUM_OPERATIONAL_SCS = 0.0;

/**
 * Volatility measurement window (cycles)
 */
export const VOLATILITY_WINDOW = 1000;

export default {
  SYSTEM_VERSION,
  GENESIS_MANDATE,
  SCS_POWER_EXPONENT,
  THRESHOLD_AVALANCHE_PROTOCOL,
  THRESHOLD_PLASTICITY_LOCK,
  THRESHOLD_POLICY_ACCEPTANCE,
  SYSTEM_HALF_LIFE,
  DECAY_LAMBDA,
  HESITATION_MULTIPLIER,
  HESITATION_MAX_CVC,
  MEMORY_CONSOLIDATION_RATIO,
  CONSOLIDATION_THRESHOLD,
  MEMORY_DECAY_K,
  MEMORY_DECAY_EPSILON,
  MIN_REPLAY_COUNT,
  AXIOMATIC_WEIGHT_FLOOR,
  COUNCIL_QUORUM_M,
  COUNCIL_TOTAL_N,
  INSTINCT_SOURCE_CONFIDENCE,
  INSTINCT_CORROBORATION_CYCLES,
  CORE_MANDATE_ATTENTION_WEIGHT,
  PATH_DIMENSION,
  GENESIS_HASH_ALGORITHM,
  SIGNATURE_ALGORITHM,
  MAX_FUNCTIONAL_HESITATION_CYCLES,
  MINIMUM_OPERATIONAL_SCS,
  VOLATILITY_WINDOW
};
