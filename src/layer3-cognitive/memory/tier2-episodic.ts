/**
 * LAYER 3: MEMORY TIER 2 - EPISODIC MEMORY (Stochastic Consolidation)
 * Patent Claims: 3, 9, 10
 * 
 * 15% Rule: TRNG identifies top 15% high-resonance frames
 * Based on emotional vector V_E (Wisdom markers)
 * Consolidates to WORM storage
 */

import { MEMORY_CONSOLIDATION_RATIO, CONSOLIDATION_THRESHOLD } from '../../config/constants';
import { ImmutableCoreMemorySystem } from '../layer1-hardware/icms';

export interface EmotionalVector {
  valence: number;      // Positive/negative (0-1)
  arousal: number;      // Intensity (0-1)
  dominance: number;    // Control (0-1)
  wisdom: number;       // Learning significance (0-1)
}

export interface EpisodicFrame {
  id: string;
  content: any;
  emotionalVector: EmotionalVector;
  importance: number;
  resonance: number;
  timestamp: number;
  consolidated: boolean;
}

export class EpisodicMemory {
  private icms: ImmutableCoreMemorySystem;
  private buffer: Map<string, EpisodicFrame>;
  private wormArchive: Map<string, EpisodicFrame>;
  private consolidationRatio: number;
  private consolidationThreshold: number;

  constructor(
    icms: ImmutableCoreMemorySystem,
    consolidationRatio: number = MEMORY_CONSOLIDATION_RATIO,
    consolidationThreshold: number = CONSOLIDATION_THRESHOLD
  ) {
    this.icms = icms;
    this.buffer = new Map();
    this.wormArchive = new Map();
    this.consolidationRatio = consolidationRatio;
    this.consolidationThreshold = consolidationThreshold;

    console.log('[EPISODIC] Initialized with 15% consolidation rule');
    console.log(`  Ratio: ${consolidationRatio}`);
    console.log(`  Threshold: ${consolidationThreshold}`);
  }

  /**
   * Patent Claim 3, 9: Store an episodic memory frame
   */
  storeFrame(content: any, emotionalVector: EmotionalVector): string {
    const id = this.generateFrameId();
    
    // Calculate importance from emotional vector
    const importance = this.calculateImportance(emotionalVector);
    
    // Calculate resonance (high-resonance = candidate for consolidation)
    const resonance = this.calculateResonance(emotionalVector);

    const frame: EpisodicFrame = {
      id,
      content,
      emotionalVector,
      importance,
      resonance,
      timestamp: Date.now(),
      consolidated: false,
    };

    this.buffer.set(id, frame);
    
    console.log(`[EPISODIC] Stored frame ${id}`);
    console.log(`  Importance: ${importance.toFixed(4)}`);
    console.log(`  Resonance: ${resonance.toFixed(4)}`);

    return id;
  }

  /**
   * Patent Claim 3, 10: 15% Stochastic Consolidation
   * Filters top 15% high-resonance frames for permanent archival
   */
  consolidate(): string[] {
    if (this.buffer.size === 0) {
      return [];
    }

    console.log(`[EPISODIC] Starting consolidation (buffer size: ${this.buffer.size})`);

    // Sort by resonance
    const frames = Array.from(this.buffer.values());
    const sortedFrames = frames.sort((a, b) => b.resonance - a.resonance);

    // Calculate 15% cutoff
    const cutoffIndex = Math.max(1, Math.floor(sortedFrames.length * this.consolidationRatio));
    const candidates = sortedFrames.slice(0, cutoffIndex);

    console.log(`[EPISODIC] Consolidating top ${cutoffIndex} frames (${(this.consolidationRatio * 100).toFixed(0)}%)`);

    const consolidated: string[] = [];

    for (const frame of candidates) {
      // Patent Claim 10: Use TRNG to determine if frame should consolidate
      // Frames above threshold have higher probability
      if (this.shouldConsolidate(frame)) {
        this.consolidateFrame(frame.id);
        consolidated.push(frame.id);
      }
    }

    // Clear buffer of consolidated frames
    for (const id of consolidated) {
      this.buffer.delete(id);
    }

    console.log(`[EPISODIC] Consolidated ${consolidated.length} frames to WORM`);
    return consolidated;
  }

  /**
   * Patent Claim 10: Stochastic consolidation decision
   * Uses TRNG for probabilistic selection
   */
  private shouldConsolidate(frame: EpisodicFrame): boolean {
    // Frames with resonance >= threshold have base probability of consolidation
    if (frame.resonance < this.consolidationThreshold) {
      return false;
    }

    // Use TRNG (True Random Number Generator simulation)
    // In production, would use hardware TRNG
    const trng = this.simulateTRNG();
    
    // Probability scales with resonance above threshold
    const excessResonance = frame.resonance - this.consolidationThreshold;
    const consolidationProbability = this.consolidationRatio + excessResonance;
    
    return trng < consolidationProbability;
  }

  /**
   * Simulate TRNG (True Random Number Generator)
   * In production, would use hardware entropy source
   */
  private simulateTRNG(): number {
    return Math.random();
  }

  /**
   * Consolidate frame into WORM storage
   */
  private consolidateFrame(frameId: string): void {
    const frame = this.buffer.get(frameId);
    
    if (!frame) {
      console.error(`[EPISODIC] Cannot consolidate unknown frame: ${frameId}`);
      return;
    }

    try {
      // Burn into ICMS WORM storage
      this.icms.burn(`EPISODIC_${frameId}`, frame);
      
      // Mark as consolidated
      frame.consolidated = true;
      
      // Move to WORM archive
      this.wormArchive.set(frameId, frame);
      
      console.log(`[EPISODIC] Frame ${frameId} consolidated to WORM`);
    } catch (error) {
      console.error(`[EPISODIC] Failed to consolidate frame ${frameId}:`, error);
    }
  }

  /**
   * Calculate importance from emotional vector
   * Patent Claim 9: Importance score influences decay rate
   */
  private calculateImportance(emotionalVector: EmotionalVector): number {
    // Weighted combination of emotional dimensions
    // Wisdom has highest weight as it indicates learning significance
    const weights = {
      valence: 0.1,
      arousal: 0.2,
      dominance: 0.1,
      wisdom: 0.6,
    };

    return (
      emotionalVector.valence * weights.valence +
      emotionalVector.arousal * weights.arousal +
      emotionalVector.dominance * weights.dominance +
      emotionalVector.wisdom * weights.wisdom
    );
  }

  /**
   * Calculate resonance (consolidation candidate score)
   */
  private calculateResonance(emotionalVector: EmotionalVector): number {
    // High arousal + high wisdom = high resonance
    return (emotionalVector.arousal + emotionalVector.wisdom) / 2;
  }

  /**
   * Retrieve a frame (from buffer or archive)
   */
  getFrame(frameId: string): EpisodicFrame | undefined {
    return this.buffer.get(frameId) || this.wormArchive.get(frameId);
  }

  /**
   * Get all frames in buffer
   */
  getBuffer(): EpisodicFrame[] {
    return Array.from(this.buffer.values());
  }

  /**
   * Get all archived frames
   */
  getArchive(): EpisodicFrame[] {
    return Array.from(this.wormArchive.values());
  }

  /**
   * Prune low-importance frames from buffer
   */
  pruneBuffer(importanceThreshold: number = 0.1): string[] {
    const pruned: string[] = [];

    for (const [id, frame] of this.buffer) {
      if (frame.importance < importanceThreshold) {
        this.buffer.delete(id);
        pruned.push(id);
      }
    }

    if (pruned.length > 0) {
      console.log(`[EPISODIC] Pruned ${pruned.length} low-importance frames`);
    }

    return pruned;
  }

  /**
   * Generate frame ID
   */
  private generateFrameId(): string {
    return `FRAME_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    bufferSize: number;
    archiveSize: number;
    consolidationRatio: number;
    averageBufferResonance: number;
    averageBufferImportance: number;
  } {
    const buffer = this.getBuffer();
    
    const averageBufferResonance = buffer.length > 0
      ? buffer.reduce((sum, f) => sum + f.resonance, 0) / buffer.length
      : 0;

    const averageBufferImportance = buffer.length > 0
      ? buffer.reduce((sum, f) => sum + f.importance, 0) / buffer.length
      : 0;

    return {
      bufferSize: this.buffer.size,
      archiveSize: this.wormArchive.size,
      consolidationRatio: this.consolidationRatio,
      averageBufferResonance,
      averageBufferImportance,
    };
  }
}

export default EpisodicMemory;
