/**
 * LAYER 2: FUNCTIONAL HESITATION (Dissonance Vector)
 * Patent Claims: 18, 19
 * 
 * Injects NOP cycles proportional to violation severity
 * Forces "hesitation" period for self-audits
 * Formula: Latency_CVC = min(50 × Cruelty_score, 500)
 */

import { HESITATION_MULTIPLIER, HESITATION_MAX_CVC } from '../../config/constants';

export interface HesitationEvent {
  timestamp: number;
  violationSeverity: number;
  cyclesInjected: number;
  reason: string;
  actualDelay: number; // milliseconds
}

export class FunctionalHesitation {
  private history: HesitationEvent[];
  private multiplier: number;
  private maxCVC: number;

  constructor(
    multiplier: number = HESITATION_MULTIPLIER,
    maxCVC: number = HESITATION_MAX_CVC
  ) {
    this.multiplier = multiplier;
    this.maxCVC = maxCVC;
    this.history = [];
  }

  /**
   * Patent Claim 18, 19: Calculate latency based on violation severity
   * Formula: Latency_CVC = min(k × severity, max_CVC)
   * 
   * @param crueltyScore - Violation severity score (0-1)
   * @param reason - Description of the violation
   * @returns Number of Cognitive Vetting Cycles (CVCs) to inject
   */
  calculateLatency(crueltyScore: number, reason: string = 'VIOLATION'): number {
    if (crueltyScore < 0 || crueltyScore > 1) {
      throw new Error(`[HESITATION] Invalid crueltyScore: ${crueltyScore} (must be in [0, 1])`);
    }

    // Calculate latency: min(k × severity, max)
    const latency = Math.min(this.multiplier * crueltyScore, this.maxCVC);
    
    console.log(`[HESITATION] Violation detected: ${reason}`);
    console.log(`[HESITATION] Severity: ${crueltyScore.toFixed(4)}`);
    console.log(`[HESITATION] Injecting ${Math.floor(latency)} Cognitive Vetting Cycles`);

    return Math.floor(latency);
  }

  /**
   * Patent Claim 18: Inject NOP cycles (hesitation period)
   * Performs redundant self-audit cycles
   * 
   * @param crueltyScore - Violation severity (0-1)
   * @param reason - Description of the violation
   */
  async inject(crueltyScore: number, reason: string = 'VIOLATION'): Promise<void> {
    const startTime = Date.now();
    const cycles = this.calculateLatency(crueltyScore, reason);

    // Inject NOP cycles - force system to "think" before acting
    for (let i = 0; i < cycles; i++) {
      // Each cycle is a self-audit operation
      await this.performSelfAudit(i, cycles);
    }

    const actualDelay = Date.now() - startTime;

    // Record event
    const event: HesitationEvent = {
      timestamp: startTime,
      violationSeverity: crueltyScore,
      cyclesInjected: cycles,
      reason,
      actualDelay,
    };
    this.history.push(event);

    console.log(`[HESITATION] Hesitation period complete: ${actualDelay}ms`);
  }

  /**
   * Perform a single self-audit cycle
   * Patent Claim 19: Forces reflection before action
   */
  private async performSelfAudit(currentCycle: number, totalCycles: number): Promise<void> {
    // Simulate cognitive vetting operation
    // In production, this would involve:
    // - Re-evaluating ethical implications
    // - Checking against ICMS mandates
    // - Verifying logical consistency
    
    // Small delay per cycle to simulate processing
    await this.sleep(1);
  }

  /**
   * Sleep for specified milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Synchronous version of inject (for non-async contexts)
   * Uses busy-wait instead of async sleep
   */
  injectSync(crueltyScore: number, reason: string = 'VIOLATION'): void {
    const startTime = Date.now();
    const cycles = this.calculateLatency(crueltyScore, reason);

    // Busy-wait for hesitation period
    const targetDelay = cycles; // 1ms per cycle
    while (Date.now() - startTime < targetDelay) {
      // Perform self-audit during busy-wait
      this.performSelfAuditSync();
    }

    const actualDelay = Date.now() - startTime;

    // Record event
    const event: HesitationEvent = {
      timestamp: startTime,
      violationSeverity: crueltyScore,
      cyclesInjected: cycles,
      reason,
      actualDelay,
    };
    this.history.push(event);
  }

  /**
   * Synchronous self-audit
   */
  private performSelfAuditSync(): void {
    // Minimal computation to simulate vetting
    let sum = 0;
    for (let i = 0; i < 100; i++) {
      sum += Math.sqrt(i);
    }
  }

  /**
   * Get hesitation history
   */
  getHistory(): readonly HesitationEvent[] {
    return this.history;
  }

  /**
   * Get total hesitation time (milliseconds)
   */
  getTotalHesitationTime(): number {
    return this.history.reduce((sum, event) => sum + event.actualDelay, 0);
  }

  /**
   * Get total cycles injected
   */
  getTotalCyclesInjected(): number {
    return this.history.reduce((sum, event) => sum + event.cyclesInjected, 0);
  }

  /**
   * Get average severity of violations
   */
  getAverageSeverity(): number {
    if (this.history.length === 0) return 0;
    const sum = this.history.reduce((acc, event) => acc + event.violationSeverity, 0);
    return sum / this.history.length;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    totalEvents: number;
    totalHesitationTime: number;
    totalCyclesInjected: number;
    averageSeverity: number;
    averageDelay: number;
  } {
    const totalHesitationTime = this.getTotalHesitationTime();
    const totalCyclesInjected = this.getTotalCyclesInjected();
    const averageSeverity = this.getAverageSeverity();
    const averageDelay = this.history.length > 0 
      ? totalHesitationTime / this.history.length 
      : 0;

    return {
      totalEvents: this.history.length,
      totalHesitationTime,
      totalCyclesInjected,
      averageSeverity,
      averageDelay,
    };
  }

  /**
   * Clear history (for testing)
   */
  clearHistory(): void {
    this.history = [];
  }

  /**
   * Check if a cruelty score requires hesitation
   */
  requiresHesitation(crueltyScore: number, threshold: number = 0.01): boolean {
    return crueltyScore > threshold;
  }

  /**
   * Get multiplier
   */
  getMultiplier(): number {
    return this.multiplier;
  }

  /**
   * Get max CVC
   */
  getMaxCVC(): number {
    return this.maxCVC;
  }
}

export default FunctionalHesitation;
