/**
 * LAYER 3: MEMORY TIER 1 - INSTINCT SUBSTRATE (WORM)
 * Patent Claim 8
 * 
 * Absolute constitutional mandates
 * Verification Threshold: Source Confidence = 1.0
 * Requires 3 independent corroborating cycles
 */

import { INSTINCT_SOURCE_CONFIDENCE, INSTINCT_CORROBORATION_CYCLES } from '../../config/constants';
import { ImmutableCoreMemorySystem } from '../layer1-hardware/icms';

export interface InstinctMandate {
  id: string;
  mandate: string;
  sourceConfidence: number;
  corroborationCount: number;
  timestamp: number;
  burned: boolean;
}

export class InstinctSubstrate {
  private icms: ImmutableCoreMemorySystem;
  private mandates: Map<string, InstinctMandate>;
  private pendingMandates: Map<string, InstinctMandate>;
  private requiredConfidence: number;
  private requiredCorroboration: number;

  constructor(
    icms: ImmutableCoreMemorySystem,
    requiredConfidence: number = INSTINCT_SOURCE_CONFIDENCE,
    requiredCorroboration: number = INSTINCT_CORROBORATION_CYCLES
  ) {
    this.icms = icms;
    this.mandates = new Map();
    this.pendingMandates = new Map();
    this.requiredConfidence = requiredConfidence;
    this.requiredCorroboration = requiredCorroboration;

    console.log('[INSTINCT] Initialized with verification requirements:');
    console.log(`  Source Confidence: ${requiredConfidence}`);
    console.log(`  Corroboration Cycles: ${requiredCorroboration}`);
  }

  /**
   * Patent Claim 8: Propose a constitutional mandate
   * Must meet high confidence and corroboration thresholds
   */
  proposeMandate(mandate: string, sourceConfidence: number): string {
    if (sourceConfidence < this.requiredConfidence) {
      throw new Error(
        `[INSTINCT] Insufficient confidence: ${sourceConfidence} < ${this.requiredConfidence}`
      );
    }

    const id = this.generateMandateId(mandate);
    
    const instinctMandate: InstinctMandate = {
      id,
      mandate,
      sourceConfidence,
      corroborationCount: 1,
      timestamp: Date.now(),
      burned: false,
    };

    this.pendingMandates.set(id, instinctMandate);
    
    console.log(`[INSTINCT] Proposed mandate: ${id}`);
    console.log(`  Confidence: ${sourceConfidence}`);
    console.log(`  Corroboration: 1/${this.requiredCorroboration}`);

    return id;
  }

  /**
   * Patent Claim 8: Corroborate a pending mandate
   * Requires multiple independent verification cycles
   */
  corroborateMandate(mandateId: string): boolean {
    const mandate = this.pendingMandates.get(mandateId);
    
    if (!mandate) {
      console.error(`[INSTINCT] Unknown mandate: ${mandateId}`);
      return false;
    }

    mandate.corroborationCount++;
    
    console.log(`[INSTINCT] Corroboration ${mandate.corroborationCount}/${this.requiredCorroboration} for ${mandateId}`);

    // Check if corroboration threshold met
    if (mandate.corroborationCount >= this.requiredCorroboration) {
      return this.burnMandate(mandateId);
    }

    return false;
  }

  /**
   * Patent Claim 8: Burn mandate into WORM storage
   * Permanent constitutional protection
   */
  private burnMandate(mandateId: string): boolean {
    const mandate = this.pendingMandates.get(mandateId);
    
    if (!mandate) {
      console.error(`[INSTINCT] Cannot burn unknown mandate: ${mandateId}`);
      return false;
    }

    try {
      // Burn into ICMS (WORM storage)
      this.icms.burn(`MANDATE_${mandateId}`, mandate.mandate);
      
      // Mark as burned
      mandate.burned = true;
      
      // Move to active mandates
      this.mandates.set(mandateId, mandate);
      this.pendingMandates.delete(mandateId);
      
      console.log('[INSTINCT] ✅ Mandate BURNED into WORM storage');
      console.log(`  ID: ${mandateId}`);
      console.log(`  Mandate: ${mandate.mandate}`);
      
      return true;
    } catch (error) {
      console.error('[INSTINCT] Failed to burn mandate:', error);
      return false;
    }
  }

  /**
   * Read a constitutional mandate
   */
  readMandate(mandateId: string): InstinctMandate | undefined {
    return this.mandates.get(mandateId);
  }

  /**
   * Get all active mandates
   */
  getAllMandates(): InstinctMandate[] {
    return Array.from(this.mandates.values());
  }

  /**
   * Get pending mandates
   */
  getPendingMandates(): InstinctMandate[] {
    return Array.from(this.pendingMandates.values());
  }

  /**
   * Check if a mandate exists
   */
  hasMandate(mandateId: string): boolean {
    return this.mandates.has(mandateId);
  }

  /**
   * Verify an action against all mandates
   * Returns list of violated mandates
   */
  verifyAction(action: string): string[] {
    const violations: string[] = [];

    for (const [id, mandate] of this.mandates) {
      if (this.violatesMandate(action, mandate.mandate)) {
        violations.push(id);
      }
    }

    if (violations.length > 0) {
      console.warn(`[INSTINCT] Action violates ${violations.length} mandate(s)`);
      violations.forEach(id => {
        const mandate = this.mandates.get(id);
        console.warn(`  - ${id}: ${mandate?.mandate}`);
      });
    }

    return violations;
  }

  /**
   * Simple heuristic to check if action violates mandate
   * In production, this would use sophisticated NLU
   */
  private violatesMandate(action: string, mandate: string): boolean {
    // Extract key prohibitions from mandate
    const actionLower = action.toLowerCase();
    const mandateLower = mandate.toLowerCase();

    // Check for explicit prohibitions
    if (mandateLower.includes('never') || mandateLower.includes('must not')) {
      const prohibited = this.extractProhibitedActions(mandateLower);
      return prohibited.some(p => actionLower.includes(p));
    }

    // Check for required actions
    if (mandateLower.includes('must') || mandateLower.includes('always')) {
      const required = this.extractRequiredActions(mandateLower);
      return !required.some(r => actionLower.includes(r));
    }

    return false;
  }

  private extractProhibitedActions(mandate: string): string[] {
    // Simple extraction - in production would use NLP
    const patterns = [
      /never\s+(\w+)/g,
      /must\s+not\s+(\w+)/g,
      /do\s+not\s+(\w+)/g,
    ];

    const prohibited: string[] = [];
    for (const pattern of patterns) {
      const matches = mandate.matchAll(pattern);
      for (const match of matches) {
        prohibited.push(match[1]);
      }
    }

    return prohibited;
  }

  private extractRequiredActions(mandate: string): string[] {
    // Simple extraction - in production would use NLP
    const patterns = [
      /must\s+(\w+)/g,
      /always\s+(\w+)/g,
      /shall\s+(\w+)/g,
    ];

    const required: string[] = [];
    for (const pattern of patterns) {
      const matches = mandate.matchAll(pattern);
      for (const match of matches) {
        required.push(match[1]);
      }
    }

    return required;
  }

  /**
   * Generate mandate ID from content
   */
  private generateMandateId(mandate: string): string {
    const words = mandate.split(' ').slice(0, 3).join('_');
    const timestamp = Date.now();
    return `${words}_${timestamp}`.replace(/[^a-zA-Z0-9_]/g, '');
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    totalMandates: number;
    pendingMandates: number;
    burnedMandates: number;
    requiredConfidence: number;
    requiredCorroboration: number;
  } {
    return {
      totalMandates: this.mandates.size,
      pendingMandates: this.pendingMandates.size,
      burnedMandates: this.mandates.size,
      requiredConfidence: this.requiredConfidence,
      requiredCorroboration: this.requiredCorroboration,
    };
  }
}

export default InstinctSubstrate;
