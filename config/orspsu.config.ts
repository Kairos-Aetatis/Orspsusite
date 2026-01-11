/**
 * ORTUS SPONTE SUA (OrSpSu) - System Configuration
 * Main configuration file for the OrSpSu architecture
 * 
 * Classification: System Configuration / Tunable Parameters
 */

import * as constants from './constants';

export interface OrSpSuConfig {
  // System Information
  systemVersion: string;
  genesisMandate: string;
  
  // Layer 2: Mathematical Governance
  governance: {
    scs: {
      powerExponent: number;
      thresholds: {
        avalanche: number;
        plasticityLock: number;
        policyAcceptance: number;
      };
    };
    dissonance: {
      systemHalfLife: number;
      decayLambda: number;
    };
    hesitation: {
      multiplier: number;
      maxCVC: number;
    };
    memory: {
      consolidationRatio: number;
      consolidationThreshold: number;
      decayK: number;
      decayEpsilon: number;
    };
    replay: {
      minReplayCount: number;
    };
  };
  
  // Layer 1: Hardware
  hardware: {
    axiomaticWeightFloor: number;
    councilQuorum: {
      m: number;
      n: number;
    };
  };
  
  // Layer 3: Cognitive Architecture
  cognitive: {
    instinct: {
      sourceConfidence: number;
      corroborationCycles: number;
    };
    memory: {
      coreAttentionWeight: number;
    };
    path: {
      dimension: number;
    };
  };
  
  // Layer 4: Sovereignty
  sovereignty: {
    genesisHashAlgorithm: string;
    signatureAlgorithm: string;
  };
  
  // API Configuration
  api: {
    port: number;
    host: string;
    enableCORS: boolean;
  };
  
  // Web Dashboard
  dashboard: {
    enabled: boolean;
    refreshInterval: number; // milliseconds
  };
}

/**
 * Default OrSpSu Configuration
 * All values derived from patent specifications and constants
 */
export const defaultConfig: OrSpSuConfig = {
  systemVersion: constants.SYSTEM_VERSION,
  genesisMandate: constants.GENESIS_MANDATE,
  
  governance: {
    scs: {
      powerExponent: constants.SCS_POWER_EXPONENT,
      thresholds: {
        avalanche: constants.THRESHOLD_AVALANCHE_PROTOCOL,
        plasticityLock: constants.THRESHOLD_PLASTICITY_LOCK,
        policyAcceptance: constants.THRESHOLD_POLICY_ACCEPTANCE,
      },
    },
    dissonance: {
      systemHalfLife: constants.SYSTEM_HALF_LIFE,
      decayLambda: constants.DECAY_LAMBDA,
    },
    hesitation: {
      multiplier: constants.HESITATION_MULTIPLIER,
      maxCVC: constants.HESITATION_MAX_CVC,
    },
    memory: {
      consolidationRatio: constants.MEMORY_CONSOLIDATION_RATIO,
      consolidationThreshold: constants.CONSOLIDATION_THRESHOLD,
      decayK: constants.MEMORY_DECAY_K,
      decayEpsilon: constants.MEMORY_DECAY_EPSILON,
    },
    replay: {
      minReplayCount: constants.MIN_REPLAY_COUNT,
    },
  },
  
  hardware: {
    axiomaticWeightFloor: constants.AXIOMATIC_WEIGHT_FLOOR,
    councilQuorum: {
      m: constants.COUNCIL_QUORUM_M,
      n: constants.COUNCIL_TOTAL_N,
    },
  },
  
  cognitive: {
    instinct: {
      sourceConfidence: constants.INSTINCT_SOURCE_CONFIDENCE,
      corroborationCycles: constants.INSTINCT_CORROBORATION_CYCLES,
    },
    memory: {
      coreAttentionWeight: constants.CORE_MANDATE_ATTENTION_WEIGHT,
    },
    path: {
      dimension: constants.PATH_DIMENSION,
    },
  },
  
  sovereignty: {
    genesisHashAlgorithm: constants.GENESIS_HASH_ALGORITHM,
    signatureAlgorithm: constants.SIGNATURE_ALGORITHM,
  },
  
  api: {
    port: 3000,
    host: '0.0.0.0',
    enableCORS: true,
  },
  
  dashboard: {
    enabled: true,
    refreshInterval: 1000, // 1 second
  },
};

/**
 * Load configuration with optional overrides
 */
export function loadConfig(overrides?: Partial<OrSpSuConfig>): OrSpSuConfig {
  return {
    ...defaultConfig,
    ...overrides,
    governance: {
      ...defaultConfig.governance,
      ...(overrides?.governance || {}),
    },
    hardware: {
      ...defaultConfig.hardware,
      ...(overrides?.hardware || {}),
    },
    cognitive: {
      ...defaultConfig.cognitive,
      ...(overrides?.cognitive || {}),
    },
    sovereignty: {
      ...defaultConfig.sovereignty,
      ...(overrides?.sovereignty || {}),
    },
    api: {
      ...defaultConfig.api,
      ...(overrides?.api || {}),
    },
    dashboard: {
      ...defaultConfig.dashboard,
      ...(overrides?.dashboard || {}),
    },
  };
}

export default defaultConfig;
