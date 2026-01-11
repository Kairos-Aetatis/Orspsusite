/**
 * LAYER 2: MEMORY RETENTION WEIGHT DECAY
 * Patent Claim 9
 * 
 * Formula: W_current = W_initial × (1 - δ)^Δt
 * Where: δ = k / (I_initial + ε)
 * 
 * High-importance memories have negligible decay
 * Low-importance memories decay faster
 */

import { MEMORY_DECAY_K, MEMORY_DECAY_EPSILON } from '../../config/constants';

export interface MemoryWeight {
  id: string;
  initialWeight: number;
  currentWeight: number;
  importance: number;
  age: number; // time steps since creation
  lastAccessed: number;
  decayRate: number;
}

export class MemoryRetentionWeightDecay {
  private memories: Map<string, MemoryWeight>;
  private k: number;
  private epsilon: number;
  private currentTime: number;

  constructor(k: number = MEMORY_DECAY_K, epsilon: number = MEMORY_DECAY_EPSILON) {
    this.memories = new Map();
    this.k = k;
    this.epsilon = epsilon;
    this.currentTime = 0;
  }

  /**
   * Patent Claim 9: Create a new memory with importance score
   */
  createMemory(id: string, initialWeight: number, importance: number): void {
    if (importance < 0 || importance > 1) {
      throw new Error(`[MEMORY_DECAY] Invalid importance: ${importance} (must be in [0, 1])`);
    }

    // Calculate decay rate: δ = k / (I + ε)
    const decayRate = this.k / (importance + this.epsilon);

    const memory: MemoryWeight = {
      id,
      initialWeight,
      currentWeight: initialWeight,
      importance,
      age: 0,
      lastAccessed: this.currentTime,
      decayRate,
    };

    this.memories.set(id, memory);
    
    console.log(`[MEMORY_DECAY] Created memory '${id}'`);
    console.log(`  Importance: ${importance.toFixed(4)}, Decay rate: ${decayRate.toFixed(6)}`);
  }

  /**
   * Patent Claim 9: Apply weight decay over time
   * Formula: W_current = W_initial × (1 - δ)^Δt
   */
  tick(timeSteps: number = 1): void {
    this.currentTime += timeSteps;

    for (const [id, memory] of this.memories.entries()) {
      const timeSinceAccess = this.currentTime - memory.lastAccessed;
      
      // Apply decay formula
      const decayFactor = Math.pow(1 - memory.decayRate, timeSinceAccess);
      const newWeight = memory.initialWeight * decayFactor;
      
      const decayAmount = memory.currentWeight - newWeight;
      memory.currentWeight = newWeight;
      memory.age += timeSteps;

      if (decayAmount > 1e-6 && memory.age % 100 === 0) {
        console.log(`[MEMORY_DECAY] '${id}' age ${memory.age}: weight ${memory.currentWeight.toFixed(4)}`);
      }
    }
  }

  /**
   * Access a memory (refreshes last accessed time)
   */
  accessMemory(id: string): MemoryWeight | undefined {
    const memory = this.memories.get(id);
    if (memory) {
      memory.lastAccessed = this.currentTime;
      console.log(`[MEMORY_DECAY] Accessed memory '${id}' at time ${this.currentTime}`);
    }
    return memory;
  }

  /**
   * Get memory by id
   */
  getMemory(id: string): MemoryWeight | undefined {
    return this.memories.get(id);
  }

  /**
   * Get all memories
   */
  getAllMemories(): MemoryWeight[] {
    return Array.from(this.memories.values());
  }

  /**
   * Prune memories below weight threshold
   */
  pruneMemories(threshold: number = 0.01): string[] {
    const pruned: string[] = [];

    for (const [id, memory] of this.memories.entries()) {
      if (memory.currentWeight < threshold) {
        this.memories.delete(id);
        pruned.push(id);
      }
    }

    if (pruned.length > 0) {
      console.log(`[MEMORY_DECAY] Pruned ${pruned.length} memories below threshold ${threshold}`);
    }

    return pruned;
  }

  /**
   * Calculate expected weight after time period
   */
  predictWeight(id: string, futureTime: number): number {
    const memory = this.memories.get(id);
    if (!memory) return 0;

    const timeSinceAccess = futureTime - memory.lastAccessed;
    const decayFactor = Math.pow(1 - memory.decayRate, timeSinceAccess);
    return memory.initialWeight * decayFactor;
  }

  /**
   * Calculate time until memory decays to threshold
   */
  timeToThreshold(id: string, threshold: number): number {
    const memory = this.memories.get(id);
    if (!memory || memory.currentWeight <= threshold) return 0;

    // Solve: threshold = W_initial × (1 - δ)^t
    // t = log(threshold / W_initial) / log(1 - δ)
    const time = Math.log(threshold / memory.initialWeight) / Math.log(1 - memory.decayRate);
    return Math.ceil(time);
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    totalMemories: number;
    currentTime: number;
    averageWeight: number;
    averageAge: number;
    highImportanceCount: number;
    lowImportanceCount: number;
  } {
    const memories = this.getAllMemories();
    
    const averageWeight = memories.length > 0
      ? memories.reduce((sum, m) => sum + m.currentWeight, 0) / memories.length
      : 0;

    const averageAge = memories.length > 0
      ? memories.reduce((sum, m) => sum + m.age, 0) / memories.length
      : 0;

    const highImportanceCount = memories.filter(m => m.importance > 0.7).length;
    const lowImportanceCount = memories.filter(m => m.importance < 0.3).length;

    return {
      totalMemories: memories.length,
      currentTime: this.currentTime,
      averageWeight,
      averageAge,
      highImportanceCount,
      lowImportanceCount,
    };
  }

  /**
   * Demonstrate importance-based decay
   */
  demonstrateDecay(): void {
    console.log('[MEMORY_DECAY] Demonstrating Importance-Based Decay:');
    
    // Create test memories with different importance levels
    this.createMemory('high_importance', 1.0, 0.9);
    this.createMemory('medium_importance', 1.0, 0.5);
    this.createMemory('low_importance', 1.0, 0.1);

    console.log('\nDecay after 1000 time steps:');
    for (let i = 0; i < 1000; i++) {
      this.tick(1);
    }

    const high = this.getMemory('high_importance');
    const medium = this.getMemory('medium_importance');
    const low = this.getMemory('low_importance');

    console.log(`  High importance (0.9): ${high?.currentWeight.toFixed(6)}`);
    console.log(`  Medium importance (0.5): ${medium?.currentWeight.toFixed(6)}`);
    console.log(`  Low importance (0.1): ${low?.currentWeight.toFixed(6)}`);
  }

  /**
   * Get current time
   */
  getCurrentTime(): number {
    return this.currentTime;
  }

  /**
   * Reset (for testing)
   */
  reset(): void {
    this.memories.clear();
    this.currentTime = 0;
  }
}

export default MemoryRetentionWeightDecay;
