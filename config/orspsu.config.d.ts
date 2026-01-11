/**
 * ORTUS SPONTE SUA (OrSpSu) - System Configuration
 * Main configuration file for the OrSpSu architecture
 *
 * Classification: System Configuration / Tunable Parameters
 */
export interface OrSpSuConfig {
    systemVersion: string;
    genesisMandate: string;
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
    hardware: {
        axiomaticWeightFloor: number;
        councilQuorum: {
            m: number;
            n: number;
        };
    };
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
    sovereignty: {
        genesisHashAlgorithm: string;
        signatureAlgorithm: string;
    };
    api: {
        port: number;
        host: string;
        enableCORS: boolean;
    };
    dashboard: {
        enabled: boolean;
        refreshInterval: number;
    };
}
/**
 * Default OrSpSu Configuration
 * All values derived from patent specifications and constants
 */
export declare const defaultConfig: OrSpSuConfig;
/**
 * Load configuration with optional overrides
 */
export declare function loadConfig(overrides?: Partial<OrSpSuConfig>): OrSpSuConfig;
export default defaultConfig;
//# sourceMappingURL=orspsu.config.d.ts.map