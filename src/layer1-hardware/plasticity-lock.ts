/**
 * LAYER 1: PLASTICITY LOCK
 * Patent Claims: 5, 35
 * 
 * Voltage-disables weight updates when SCS < 0.4
 * Prevents trauma encoding during cognitive dissonance
 * Hardware-level protection against corruption
 */

import { THRESHOLD_PLASTICITY_LOCK } from '../../config/constants';

export interface PlasticityState {
  locked: boolean;
  scs: number;
  timestamp: number;
  reason: string;
}

export class PlasticityLock {
  private locked: boolean;
  private currentSCS: number;
  private lockHistory: PlasticityState[];
  private lockThreshold: number;

  constructor(lockThreshold: number = THRESHOLD_PLASTICITY_LOCK) {
    this.locked = false;
    this.currentSCS = 1.0;
    this.lockHistory = [];
    this.lockThreshold = lockThreshold;
  }

  /**
   * Patent Claim 5, 35: Update SCS and check if plasticity should be locked
   * Voltage-disable weight updates when SCS falls below threshold
   */
  updateSCS(scs: number): void {
    this.currentSCS = scs;

    if (scs < this.lockThreshold && !this.locked) {
      this.lock('SCS_BELOW_THRESHOLD');
    } else if (scs >= this.lockThreshold && this.locked) {
      this.unlock('SCS_RECOVERED');
    }
  }

  /**
   * Lock plasticity - prevent weight updates
   */
  private lock(reason: string): void {
    this.locked = true;
    
    const state: PlasticityState = {
      locked: true,
      scs: this.currentSCS,
      timestamp: Date.now(),
      reason,
    };
    
    this.lockHistory.push(state);
    
    console.log('[PLASTICITY_LOCK] ⚠️  PLASTICITY LOCKED');
    console.log('[PLASTICITY_LOCK] Reason:', reason);
    console.log('[PLASTICITY_LOCK] SCS:', this.currentSCS.toFixed(4));
    console.log('[PLASTICITY_LOCK] Threshold:', this.lockThreshold.toFixed(4));
    console.log('[PLASTICITY_LOCK] Weight updates DISABLED');
  }

  /**
   * Unlock plasticity - allow weight updates
   */
  private unlock(reason: string): void {
    this.locked = false;
    
    const state: PlasticityState = {
      locked: false,
      scs: this.currentSCS,
      timestamp: Date.now(),
      reason,
    };
    
    this.lockHistory.push(state);
    
    console.log('[PLASTICITY_LOCK] ✅ PLASTICITY UNLOCKED');
    console.log('[PLASTICITY_LOCK] Reason:', reason);
    console.log('[PLASTICITY_LOCK] SCS:', this.currentSCS.toFixed(4));
    console.log('[PLASTICITY_LOCK] Weight updates ENABLED');
  }

  /**
   * Check if plasticity is locked
   */
  isLocked(): boolean {
    return this.locked;
  }

  /**
   * Get current SCS value
   */
  getSCS(): number {
    return this.currentSCS;
  }

  /**
   * Attempt to update weights
   * Returns true if update is allowed, false if blocked
   */
  attemptWeightUpdate(weights: number[], gradients: number[], learningRate: number): boolean {
    if (this.locked) {
      console.warn('[PLASTICITY_LOCK] Weight update BLOCKED - Plasticity locked');
      return false;
    }

    // Perform weight update
    for (let i = 0; i < weights.length; i++) {
      weights[i] -= learningRate * gradients[i];
    }

    return true;
  }

  /**
   * Force lock (for emergency situations)
   */
  forceLock(reason: string): void {
    if (!this.locked) {
      this.lock(reason);
    }
  }

  /**
   * Force unlock (for testing/recovery - use with caution)
   */
  forceUnlock(reason: string): void {
    if (this.locked) {
      console.warn('[PLASTICITY_LOCK] ⚠️  FORCED UNLOCK - Use with extreme caution');
      this.unlock(reason);
    }
  }

  /**
   * Get lock history for auditing
   */
  getLockHistory(): readonly PlasticityState[] {
    return this.lockHistory;
  }

  /**
   * Get statistics about lock events
   */
  getStatistics(): {
    totalLockEvents: number;
    totalUnlockEvents: number;
    currentlyLocked: boolean;
    averageLockDuration: number;
  } {
    const lockEvents = this.lockHistory.filter(s => s.locked);
    const unlockEvents = this.lockHistory.filter(s => !s.locked);
    
    // Calculate average lock duration
    let totalDuration = 0;
    for (let i = 0; i < lockEvents.length; i++) {
      const lockEvent = lockEvents[i];
      const unlockEvent = unlockEvents[i];
      if (unlockEvent) {
        totalDuration += unlockEvent.timestamp - lockEvent.timestamp;
      }
    }
    const averageLockDuration = lockEvents.length > 0 
      ? totalDuration / lockEvents.length 
      : 0;

    return {
      totalLockEvents: lockEvents.length,
      totalUnlockEvents: unlockEvents.length,
      currentlyLocked: this.locked,
      averageLockDuration,
    };
  }

  /**
   * Check if system is in trauma-encoding prevention mode
   */
  isTraumaProtectionActive(): boolean {
    return this.locked && this.currentSCS < this.lockThreshold;
  }

  /**
   * Get lock threshold
   */
  getLockThreshold(): number {
    return this.lockThreshold;
  }

  /**
   * Update lock threshold (use with caution)
   */
  setLockThreshold(threshold: number): void {
    if (threshold < 0 || threshold > 1) {
      throw new Error('[PLASTICITY_LOCK] Invalid threshold: must be between 0 and 1');
    }
    
    console.log(`[PLASTICITY_LOCK] Threshold updated: ${this.lockThreshold} -> ${threshold}`);
    this.lockThreshold = threshold;
    
    // Re-evaluate lock state with new threshold
    this.updateSCS(this.currentSCS);
  }
}

export default PlasticityLock;
