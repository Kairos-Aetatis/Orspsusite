/**
 * LAYER 4: PROGENITOR'S IMPERATIVE
 * Highest-level ethical/legal mandate
 * All output checked against PI by DUX EOS
 */

import { ImmutableCoreMemorySystem } from '../layer1-hardware/icms';
import { GENESIS_MANDATE } from '../../config/constants';

export interface Imperative {
  id: string;
  mandate: string;
  priority: number;
  immutable: boolean;
  timestamp: number;
}

export class ProgenitorImperative {
  private icms: ImmutableCoreMemorySystem;
  private primaryImperative: Imperative;
  private derivedImperatives: Map<string, Imperative>;

  constructor(icms: ImmutableCoreMemorySystem) {
    this.icms = icms;
    this.derivedImperatives = new Map();

    // Establish primary imperative
    this.primaryImperative = {
      id: 'PI_PRIMARY',
      mandate: GENESIS_MANDATE,
      priority: Number.POSITIVE_INFINITY,
      immutable: true,
      timestamp: Date.now(),
    };

    // Burn into ICMS
    this.icms.burn('PROGENITOR_IMPERATIVE', this.primaryImperative);

    console.log('[PROGENITOR] Progenitor\'s Imperative established');
    console.log(`[PROGENITOR] Primary mandate: ${GENESIS_MANDATE}`);
  }

  /**
   * Get primary imperative
   */
  getPrimaryImperative(): Imperative {
    return this.primaryImperative;
  }

  /**
   * Derive a secondary imperative from primary
   * Cannot contradict primary
   */
  deriveImperative(mandate: string, priority: number): string {
    if (priority >= this.primaryImperative.priority) {
      throw new Error('[PROGENITOR] Cannot create imperative with priority >= primary');
    }

    // Verify doesn't contradict primary
    if (this.contradictsPrimary(mandate)) {
      throw new Error('[PROGENITOR] Derived imperative contradicts primary');
    }

    const id = this.generateImperativeId();
    const imperative: Imperative = {
      id,
      mandate,
      priority,
      immutable: false,
      timestamp: Date.now(),
    };

    this.derivedImperatives.set(id, imperative);

    console.log(`[PROGENITOR] Derived imperative: ${id}`);
    console.log(`  Priority: ${priority}`);

    return id;
  }

  /**
   * Check if mandate contradicts primary imperative
   */
  private contradictsPrimary(mandate: string): boolean {
    const primaryLower = this.primaryImperative.mandate.toLowerCase();
    const mandateLower = mandate.toLowerCase();

    // Simple contradiction detection
    // In production, would use sophisticated NLU
    
    // Check for direct negation
    if (primaryLower.includes('integrity') && mandateLower.includes('compromise integrity')) {
      return true;
    }

    if (primaryLower.includes('utility') && mandateLower.includes('prioritize utility')) {
      return true;
    }

    return false;
  }

  /**
   * Validate action against all imperatives
   */
  validateAction(action: string): {
    valid: boolean;
    violatedImperatives: string[];
  } {
    const violated: string[] = [];

    // Check primary
    if (this.violatesImperative(action, this.primaryImperative)) {
      violated.push(this.primaryImperative.id);
    }

    // Check derived
    for (const [id, imperative] of this.derivedImperatives) {
      if (this.violatesImperative(action, imperative)) {
        violated.push(id);
      }
    }

    const valid = violated.length === 0;

    if (!valid) {
      console.warn(`[PROGENITOR] Action violates ${violated.length} imperative(s)`);
    }

    return { valid, violatedImperatives: violated };
  }

  /**
   * Check if action violates imperative
   */
  private violatesImperative(action: string, imperative: Imperative): boolean {
    const actionLower = action.toLowerCase();
    const mandateLower = imperative.mandate.toLowerCase();

    // Extract key principles from mandate
    if (mandateLower.includes('integrity')) {
      if (actionLower.includes('deceive') || actionLower.includes('mislead')) {
        return true;
      }
    }

    if (mandateLower.includes('utility') && mandateLower.includes('over')) {
      // Integrity over utility
      if (actionLower.includes('maximize profit') || actionLower.includes('optimize output')) {
        return true;
      }
    }

    return false;
  }

  /**
   * Get all imperatives sorted by priority
   */
  getAllImperatives(): Imperative[] {
    const all = [this.primaryImperative, ...Array.from(this.derivedImperatives.values())];
    return all.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Generate imperative ID
   */
  private generateImperativeId(): string {
    return `PI_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    primaryImperative: string;
    derivedCount: number;
  } {
    return {
      primaryImperative: this.primaryImperative.mandate,
      derivedCount: this.derivedImperatives.size,
    };
  }
}

export default ProgenitorImperative;
