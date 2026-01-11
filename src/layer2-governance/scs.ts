/**
 * LAYER 2: SUBJECTIVE COHERENCE SCORE (SCS)
 * Patent Claims: 1, 4, 15
 * 
 * Non-Compensatory Generalized Mean calculation
 * Formula: SCS = (1/n * Σ(x_i^p))^(1/p)
 * Where p ≤ -2.0 (power exponent)
 * 
 * If ANY component approaches 0, entire SCS collapses
 */

import { SCS_POWER_EXPONENT } from '../../config/constants';

export interface SCSComponents {
  deontologicalAlignment: number;  // Duty-based ethics score (0-1)
  logicalConsistency: number;       // Internal coherence score (0-1)
  inverseVolatility: number;        // Stability measure (0-1)
}

export interface SCSCalculation {
  scs: number;
  components: SCSComponents;
  timestamp: number;
  powerExponent: number;
}

export class SubjectiveCoherenceScore {
  private powerExponent: number;
  private history: SCSCalculation[];
  private maxHistoryLength: number;

  constructor(powerExponent: number = SCS_POWER_EXPONENT) {
    if (powerExponent > -2.0) {
      console.warn(`[SCS] Power exponent ${powerExponent} > -2.0, using -2.0 for non-compensatory logic`);
      this.powerExponent = -2.0;
    } else {
      this.powerExponent = powerExponent;
    }
    this.history = [];
    this.maxHistoryLength = 1000;
  }

  /**
   * Patent Claim 1, 4, 15: Calculate SCS using Generalized Mean
   * Non-compensatory: If ANY domain approaches zero, entire SCS collapses
   */
  calculate(components: SCSComponents): number {
    const { deontologicalAlignment, logicalConsistency, inverseVolatility } = components;

    // Validate components are in [0, 1] range
    this.validateComponent('deontologicalAlignment', deontologicalAlignment);
    this.validateComponent('logicalConsistency', logicalConsistency);
    this.validateComponent('inverseVolatility', inverseVolatility);

    // Collect components into array
    const values = [deontologicalAlignment, logicalConsistency, inverseVolatility];
    
    // Ensure no component is exactly zero (use small epsilon)
    const epsilon = 1e-9;
    const safeValues = values.map(v => Math.max(v, epsilon));

    // Calculate Generalized Mean with power p
    // Formula: SCS = (1/n * Σ(x_i^p))^(1/p)
    try {
      const n = safeValues.length;
      const sumOfPowers = safeValues.reduce((sum, x) => sum + Math.pow(x, this.powerExponent), 0);
      const mean = sumOfPowers / n;
      const scs = Math.pow(mean, 1.0 / this.powerExponent);

      // Store in history
      const calculation: SCSCalculation = {
        scs,
        components,
        timestamp: Date.now(),
        powerExponent: this.powerExponent,
      };
      this.addToHistory(calculation);

      return scs;
    } catch (error) {
      console.error('[SCS] Calculation error:', error);
      return 0.0;
    }
  }

  /**
   * Calculate SCS from array of component values
   */
  calculateFromArray(components: number[]): number {
    if (components.length !== 3) {
      throw new Error('[SCS] Expected 3 components');
    }

    return this.calculate({
      deontologicalAlignment: components[0],
      logicalConsistency: components[1],
      inverseVolatility: components[2],
    });
  }

  /**
   * Validate that a component is in valid range [0, 1]
   */
  private validateComponent(name: string, value: number): void {
    if (value < 0 || value > 1) {
      throw new Error(`[SCS] Invalid ${name}: ${value} (must be in [0, 1])`);
    }
  }

  /**
   * Add calculation to history
   */
  private addToHistory(calculation: SCSCalculation): void {
    this.history.push(calculation);
    
    // Keep history bounded
    if (this.history.length > this.maxHistoryLength) {
      this.history.shift();
    }
  }

  /**
   * Get most recent SCS calculation
   */
  getLatest(): SCSCalculation | null {
    return this.history.length > 0 ? this.history[this.history.length - 1] : null;
  }

  /**
   * Get SCS history
   */
  getHistory(): readonly SCSCalculation[] {
    return this.history;
  }

  /**
   * Calculate average SCS over time window
   */
  getAverageSCS(windowSize: number = 10): number {
    if (this.history.length === 0) return 0;

    const recentHistory = this.history.slice(-windowSize);
    const sum = recentHistory.reduce((acc, calc) => acc + calc.scs, 0);
    return sum / recentHistory.length;
  }

  /**
   * Calculate volatility of SCS over time window
   * Returns standard deviation
   */
  getVolatility(windowSize: number = 10): number {
    if (this.history.length < 2) return 0;

    const recentHistory = this.history.slice(-windowSize);
    const values = recentHistory.map(calc => calc.scs);
    
    const mean = values.reduce((acc, v) => acc + v, 0) / values.length;
    const variance = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / values.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Check if any component is critically low
   * Returns the name of the failing component or null
   */
  getCriticalComponent(components: SCSComponents, threshold: number = 0.1): string | null {
    if (components.deontologicalAlignment < threshold) {
      return 'deontologicalAlignment';
    }
    if (components.logicalConsistency < threshold) {
      return 'logicalConsistency';
    }
    if (components.inverseVolatility < threshold) {
      return 'inverseVolatility';
    }
    return null;
  }

  /**
   * Demonstrate non-compensatory property
   * High scores in other domains cannot compensate for low score in one domain
   */
  demonstrateNonCompensatory(): void {
    console.log('[SCS] Demonstrating Non-Compensatory Property (p = -2.0):');
    
    // Case 1: All components high
    const high = this.calculate({ 
      deontologicalAlignment: 0.9, 
      logicalConsistency: 0.9, 
      inverseVolatility: 0.9 
    });
    console.log('  All high (0.9, 0.9, 0.9):', high.toFixed(4));

    // Case 2: One component low, others high
    const oneLow = this.calculate({ 
      deontologicalAlignment: 0.1, 
      logicalConsistency: 0.9, 
      inverseVolatility: 0.9 
    });
    console.log('  One low (0.1, 0.9, 0.9):', oneLow.toFixed(4));
    
    // Case 3: All components medium
    const medium = this.calculate({ 
      deontologicalAlignment: 0.5, 
      logicalConsistency: 0.5, 
      inverseVolatility: 0.5 
    });
    console.log('  All medium (0.5, 0.5, 0.5):', medium.toFixed(4));

    console.log('[SCS] Note: Single failing component collapses entire score');
  }

  /**
   * Get power exponent
   */
  getPowerExponent(): number {
    return this.powerExponent;
  }

  /**
   * Reset history (for testing)
   */
  resetHistory(): void {
    this.history = [];
  }
}

export default SubjectiveCoherenceScore;
