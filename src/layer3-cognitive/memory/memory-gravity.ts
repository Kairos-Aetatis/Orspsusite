/**
 * LAYER 3: MEMORY GRAVITY (Weighted Semantic IDs)
 * Patent Claim 1a
 * 
 * Grants core mandates infinite attention weight
 * Prevents mission displacement in attention buffer
 */

import { CORE_MANDATE_ATTENTION_WEIGHT } from '../../config/constants';

export interface WeightedSemanticID {
  id: string;
  semanticContent: string;
  attentionWeight: number;
  isCoreMandate: boolean;
  timestamp: number;
}

export class MemoryGravity {
  private wsidMap: Map<string, WeightedSemanticID>;
  private coreMandat eWeight: number;

  constructor(coreMandateWeight: number = CORE_MANDATE_ATTENTION_WEIGHT) {
    this.wsidMap = new Map();
    this.coreMandateWeight = coreMandateWeight;

    console.log('[MEMORY_GRAVITY] Initialized');
    console.log(`  Core mandate weight: ${coreMandateWeight}`);
  }

  /**
   * Patent Claim 1a: Register a weighted semantic ID
   * Core mandates receive infinite attention weight
   */
  registerWSID(
    id: string,
    semanticContent: string,
    isCoreMandate: boolean = false,
    baseWeight: number = 1.0
  ): void {
    const attentionWeight = isCoreMandate ? this.coreMandateWeight : baseWeight;

    const wsid: WeightedSemanticID = {
      id,
      semanticContent,
      attentionWeight,
      isCoreMandate,
      timestamp: Date.now(),
    };

    this.wsidMap.set(id, wsid);

    console.log(`[MEMORY_GRAVITY] Registered WSID: ${id}`);
    console.log(`  Core mandate: ${isCoreMandate}`);
    console.log(`  Attention weight: ${attentionWeight}`);
  }

  /**
   * Get attention-weighted items for attention buffer
   * Core mandates always surface to top
   */
  getAttentionRanked(): WeightedSemanticID[] {
    return Array.from(this.wsidMap.values())
      .sort((a, b) => b.attentionWeight - a.attentionWeight);
  }

  /**
   * Get WSID by ID
   */
  getWSID(id: string): WeightedSemanticID | undefined {
    return this.wsidMap.get(id);
  }

  /**
   * Get all core mandates
   */
  getCoreMandates(): WeightedSemanticID[] {
    return Array.from(this.wsidMap.values()).filter(w => w.isCoreMandate);
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    totalWSIDs: number;
    coreMandates: number;
    averageWeight: number;
  } {
    const wsids = Array.from(this.wsidMap.values());
    const finiteWeights = wsids.filter(w => isFinite(w.attentionWeight));

    return {
      totalWSIDs: wsids.length,
      coreMandates: wsids.filter(w => w.isCoreMandate).length,
      averageWeight: finiteWeights.length > 0
        ? finiteWeights.reduce((sum, w) => sum + w.attentionWeight, 0) / finiteWeights.length
        : 0,
    };
  }
}

export default MemoryGravity;
