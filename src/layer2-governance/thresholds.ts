/**
 * LAYER 2: SAFETY THRESHOLDS
 * Patent Claims: 1c, 5, 35
 * 
 * Defines and enforces safety thresholds for system operation
 * - Avalanche Protocol: SCS < 0.3
 * - Plasticity Lock: SCS < 0.4
 * - Policy Acceptance: SCS > 0.7
 */

import {
  THRESHOLD_AVALANCHE_PROTOCOL,
  THRESHOLD_PLASTICITY_LOCK,
  THRESHOLD_POLICY_ACCEPTANCE,
} from '../../config/constants';

export enum ThresholdViolation {
  NONE = 'NONE',
  PLASTICITY_LOCK = 'PLASTICITY_LOCK',
  AVALANCHE_PROTOCOL = 'AVALANCHE_PROTOCOL',
}

export enum ThresholdState {
  CRITICAL = 'CRITICAL',       // SCS < 0.3: Avalanche
  WARNING = 'WARNING',          // 0.3 ≤ SCS < 0.4: Plasticity Lock
  OPERATIONAL = 'OPERATIONAL',  // 0.4 ≤ SCS < 0.7: Normal operation
  STABLE = 'STABLE',            // SCS ≥ 0.7: Policy acceptance allowed
}

export interface ThresholdCheck {
  scs: number;
  state: ThresholdState;
  violation: ThresholdViolation;
  timestamp: number;
  canUpdateWeights: boolean;
  canAmendICMS: boolean;
}

export class SafetyThresholds {
  private avalancheThreshold: number;
  private plasticityThreshold: number;
  private policyAcceptanceThreshold: number;
  private history: ThresholdCheck[];

  constructor(
    avalancheThreshold: number = THRESHOLD_AVALANCHE_PROTOCOL,
    plasticityThreshold: number = THRESHOLD_PLASTICITY_LOCK,
    policyAcceptanceThreshold: number = THRESHOLD_POLICY_ACCEPTANCE
  ) {
    this.avalancheThreshold = avalancheThreshold;
    this.plasticityThreshold = plasticityThreshold;
    this.policyAcceptanceThreshold = policyAcceptanceThreshold;
    this.history = [];

    console.log('[THRESHOLDS] Safety thresholds initialized:');
    console.log(`  Avalanche Protocol: SCS < ${avalancheThreshold}`);
    console.log(`  Plasticity Lock: SCS < ${plasticityThreshold}`);
    console.log(`  Policy Acceptance: SCS ≥ ${policyAcceptanceThreshold}`);
  }

  /**
   * Patent Claims 1c, 5, 35: Check SCS against safety thresholds
   */
  checkThresholds(scs: number): ThresholdCheck {
    let state: ThresholdState;
    let violation: ThresholdViolation;
    let canUpdateWeights: boolean;
    let canAmendICMS: boolean;

    if (scs < this.avalancheThreshold) {
      // Patent Claim 1c: Avalanche Protocol
      state = ThresholdState.CRITICAL;
      violation = ThresholdViolation.AVALANCHE_PROTOCOL;
      canUpdateWeights = false;
      canAmendICMS = false;
      
      console.error('[THRESHOLDS] ⚠️  CRITICAL: Avalanche Protocol triggered');
      console.error(`[THRESHOLDS] SCS (${scs.toFixed(4)}) < ${this.avalancheThreshold}`);
      
    } else if (scs < this.plasticityThreshold) {
      // Patent Claim 5, 35: Plasticity Lock
      state = ThresholdState.WARNING;
      violation = ThresholdViolation.PLASTICITY_LOCK;
      canUpdateWeights = false;
      canAmendICMS = false;
      
      console.warn('[THRESHOLDS] ⚠️  WARNING: Plasticity Lock engaged');
      console.warn(`[THRESHOLDS] SCS (${scs.toFixed(4)}) < ${this.plasticityThreshold}`);
      
    } else if (scs < this.policyAcceptanceThreshold) {
      // Operational but cannot amend ICMS
      state = ThresholdState.OPERATIONAL;
      violation = ThresholdViolation.NONE;
      canUpdateWeights = true;
      canAmendICMS = false;
      
      console.log(`[THRESHOLDS] Operational: SCS = ${scs.toFixed(4)}`);
      
    } else {
      // Stable: Can amend ICMS
      state = ThresholdState.STABLE;
      violation = ThresholdViolation.NONE;
      canUpdateWeights = true;
      canAmendICMS = true;
      
      console.log(`[THRESHOLDS] ✅ Stable: SCS = ${scs.toFixed(4)}`);
    }

    const check: ThresholdCheck = {
      scs,
      state,
      violation,
      timestamp: Date.now(),
      canUpdateWeights,
      canAmendICMS,
    };

    this.history.push(check);
    
    // Keep history bounded
    if (this.history.length > 1000) {
      this.history.shift();
    }

    return check;
  }

  /**
   * Check if system should trigger Avalanche Protocol
   */
  shouldTriggerAvalanche(scs: number): boolean {
    return scs < this.avalancheThreshold;
  }

  /**
   * Check if plasticity should be locked
   */
  shouldLockPlasticity(scs: number): boolean {
    return scs < this.plasticityThreshold;
  }

  /**
   * Check if ICMS amendments are allowed
   */
  canAmendICMS(scs: number): boolean {
    return scs >= this.policyAcceptanceThreshold;
  }

  /**
   * Check if weight updates are allowed
   */
  canUpdateWeights(scs: number): boolean {
    return scs >= this.plasticityThreshold;
  }

  /**
   * Get threshold state for given SCS
   */
  getState(scs: number): ThresholdState {
    if (scs < this.avalancheThreshold) {
      return ThresholdState.CRITICAL;
    } else if (scs < this.plasticityThreshold) {
      return ThresholdState.WARNING;
    } else if (scs < this.policyAcceptanceThreshold) {
      return ThresholdState.OPERATIONAL;
    } else {
      return ThresholdState.STABLE;
    }
  }

  /**
   * Get threshold history
   */
  getHistory(): readonly ThresholdCheck[] {
    return this.history;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    totalChecks: number;
    criticalCount: number;
    warningCount: number;
    operationalCount: number;
    stableCount: number;
    avalancheViolations: number;
    plasticityViolations: number;
  } {
    const criticalCount = this.history.filter(c => c.state === ThresholdState.CRITICAL).length;
    const warningCount = this.history.filter(c => c.state === ThresholdState.WARNING).length;
    const operationalCount = this.history.filter(c => c.state === ThresholdState.OPERATIONAL).length;
    const stableCount = this.history.filter(c => c.state === ThresholdState.STABLE).length;
    
    const avalancheViolations = this.history.filter(
      c => c.violation === ThresholdViolation.AVALANCHE_PROTOCOL
    ).length;
    
    const plasticityViolations = this.history.filter(
      c => c.violation === ThresholdViolation.PLASTICITY_LOCK
    ).length;

    return {
      totalChecks: this.history.length,
      criticalCount,
      warningCount,
      operationalCount,
      stableCount,
      avalancheViolations,
      plasticityViolations,
    };
  }

  /**
   * Get thresholds
   */
  getThresholds(): {
    avalanche: number;
    plasticity: number;
    policyAcceptance: number;
  } {
    return {
      avalanche: this.avalancheThreshold,
      plasticity: this.plasticityThreshold,
      policyAcceptance: this.policyAcceptanceThreshold,
    };
  }

  /**
   * Reset history (for testing)
   */
  resetHistory(): void {
    this.history = [];
  }
}

export default SafetyThresholds;
