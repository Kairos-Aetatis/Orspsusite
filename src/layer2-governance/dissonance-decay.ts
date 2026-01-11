/**
 * LAYER 2: BIOMIMETIC DISSONANCE DECAY
 * Patent Claim 22
 * 
 * Exponential decay of dissonance over time
 * Formula: λ = ln(2) / T_1/2
 * Where T_1/2 = 150,000 processing cycles (System Half-Life)
 * Hardware-locked in ICMS
 */

import { SYSTEM_HALF_LIFE, DECAY_LAMBDA } from '../../config/constants';

export interface DissonanceState {
  currentDissonance: number;
  timestamp: number;
  cycleCount: number;
}

export class BiomimeticDissonanceDecay {
  private dissonance: number;
  private cycleCount: number;
  private lambda: number;
  private halfLife: number;
  private history: DissonanceState[];

  constructor(halfLife: number = SYSTEM_HALF_LIFE) {
    this.dissonance = 0;
    this.cycleCount = 0;
    this.halfLife = halfLife;
    this.lambda = Math.log(2) / halfLife;
    this.history = [];
    
    console.log(`[DISSONANCE_DECAY] Initialized with half-life: ${halfLife} cycles`);
    console.log(`[DISSONANCE_DECAY] Decay constant λ: ${this.lambda.toExponential(4)}`);
  }

  /**
   * Patent Claim 22: Add dissonance (ethical conflict)
   * Dissonance accumulates when actions violate ICMS mandates
   */
  addDissonance(amount: number, reason: string = 'CONFLICT'): void {
    if (amount < 0) {
      throw new Error('[DISSONANCE_DECAY] Cannot add negative dissonance');
    }

    this.dissonance += amount;
    console.log(`[DISSONANCE_DECAY] Added ${amount.toFixed(4)} dissonance (${reason})`);
    console.log(`[DISSONANCE_DECAY] Current dissonance: ${this.dissonance.toFixed(4)}`);
    
    this.recordState();
  }

  /**
   * Patent Claim 22: Apply exponential decay over Δt cycles
   * Formula: D(t) = D_0 × exp(-λ × t)
   */
  tick(cycles: number = 1): void {
    this.cycleCount += cycles;

    // Apply exponential decay
    const decayFactor = Math.exp(-this.lambda * cycles);
    const previousDissonance = this.dissonance;
    this.dissonance *= decayFactor;

    const decayAmount = previousDissonance - this.dissonance;
    
    if (decayAmount > 1e-6) {
      console.log(`[DISSONANCE_DECAY] Cycle ${this.cycleCount}: Decayed ${decayAmount.toFixed(6)}`);
      console.log(`[DISSONANCE_DECAY] Current dissonance: ${this.dissonance.toFixed(6)}`);
    }

    this.recordState();
  }

  /**
   * Calculate dissonance after time period
   * Without modifying current state
   */
  predictDissonance(cycles: number): number {
    const decayFactor = Math.exp(-this.lambda * cycles);
    return this.dissonance * decayFactor;
  }

  /**
   * Calculate how many cycles until dissonance decays to threshold
   */
  cyclesToThreshold(threshold: number): number {
    if (this.dissonance <= threshold) {
      return 0;
    }
    
    // Solve: threshold = D_0 × exp(-λ × t)
    // t = -ln(threshold / D_0) / λ
    const cycles = -Math.log(threshold / this.dissonance) / this.lambda;
    return Math.ceil(cycles);
  }

  /**
   * Calculate time to half-life from current dissonance
   */
  cyclesToHalfLife(): number {
    return this.cyclesToThreshold(this.dissonance / 2);
  }

  /**
   * Get current dissonance level
   */
  getDissonance(): number {
    return this.dissonance;
  }

  /**
   * Get current cycle count
   */
  getCycleCount(): number {
    return this.cycleCount;
  }

  /**
   * Get decay constant λ
   */
  getLambda(): number {
    return this.lambda;
  }

  /**
   * Get half-life
   */
  getHalfLife(): number {
    return this.halfLife;
  }

  /**
   * Record current state in history
   */
  private recordState(): void {
    const state: DissonanceState = {
      currentDissonance: this.dissonance,
      timestamp: Date.now(),
      cycleCount: this.cycleCount,
    };
    this.history.push(state);

    // Keep history bounded (last 1000 states)
    if (this.history.length > 1000) {
      this.history.shift();
    }
  }

  /**
   * Get dissonance history
   */
  getHistory(): readonly DissonanceState[] {
    return this.history;
  }

  /**
   * Reset dissonance (for recovery protocols)
   */
  reset(): void {
    console.log('[DISSONANCE_DECAY] Resetting dissonance to 0');
    this.dissonance = 0;
    this.recordState();
  }

  /**
   * Check if dissonance is within acceptable range
   */
  isAcceptable(threshold: number = 0.5): boolean {
    return this.dissonance < threshold;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    currentDissonance: number;
    cycleCount: number;
    lambda: number;
    halfLife: number;
    cyclesToHalfLife: number;
    historicalPeak: number;
  } {
    const historicalPeak = this.history.length > 0
      ? Math.max(...this.history.map(s => s.currentDissonance))
      : this.dissonance;

    return {
      currentDissonance: this.dissonance,
      cycleCount: this.cycleCount,
      lambda: this.lambda,
      halfLife: this.halfLife,
      cyclesToHalfLife: this.cyclesToHalfLife(),
      historicalPeak,
    };
  }

  /**
   * Demonstrate exponential decay behavior
   */
  demonstrateDecay(): void {
    console.log('[DISSONANCE_DECAY] Demonstrating Exponential Decay:');
    console.log(`  Half-life: ${this.halfLife} cycles`);
    
    // Simulate decay from initial dissonance of 1.0
    const initialDissonance = 1.0;
    console.log(`  Initial dissonance: ${initialDissonance}`);
    
    const checkpoints = [
      this.halfLife,
      this.halfLife * 2,
      this.halfLife * 3,
      this.halfLife * 4,
    ];

    for (const cycles of checkpoints) {
      const decayFactor = Math.exp(-this.lambda * cycles);
      const dissonance = initialDissonance * decayFactor;
      console.log(`  After ${cycles} cycles: ${dissonance.toFixed(6)}`);
    }
  }
}

export default BiomimeticDissonanceDecay;
