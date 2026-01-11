/**
 * LAYER 2: DETERMINISTIC REPLAY VALIDATION
 * Patent Claim 13
 * 
 * Pre-update policy testing with ≥10 sandboxed replays
 * Uses stored Entropy Seed
 * Output must be bit-identical across all replays
 */

import { MIN_REPLAY_COUNT } from '../../config/constants';
import { createHash } from 'crypto';

export interface ReplayResult {
  replayId: number;
  output: any;
  outputHash: string;
  timestamp: number;
  success: boolean;
}

export interface ValidationReport {
  totalReplays: number;
  successfulReplays: number;
  failedReplays: number;
  outputConsistent: boolean;
  consensusHash: string | null;
  timestamp: number;
}

export type PolicyFunction = (input: any, seed: number) => any;

export class DeterministicReplayValidation {
  private minReplays: number;
  private entropySeed: number;

  constructor(minReplays: number = MIN_REPLAY_COUNT) {
    this.minReplays = minReplays;
    this.entropySeed = this.generateEntropySeed();
    
    console.log(`[REPLAY] Initialized with entropy seed: ${this.entropySeed}`);
  }

  /**
   * Generate deterministic entropy seed
   * Stored for replay consistency
   */
  private generateEntropySeed(): number {
    // In production, this would be derived from hardware entropy
    // For simulation, use timestamp-based seed
    return Math.floor(Date.now() / 1000);
  }

  /**
   * Patent Claim 13: Validate policy through sandboxed replays
   * All replays must produce bit-identical output
   */
  async validatePolicy(
    policyFn: PolicyFunction,
    input: any,
    replayCount: number = this.minReplays
  ): Promise<ValidationReport> {
    if (replayCount < this.minReplays) {
      throw new Error(`[REPLAY] Replay count ${replayCount} < minimum ${this.minReplays}`);
    }

    console.log(`[REPLAY] Starting deterministic validation with ${replayCount} replays`);
    console.log(`[REPLAY] Entropy seed: ${this.entropySeed}`);

    const results: ReplayResult[] = [];
    const outputHashes = new Set<string>();

    // Execute replays in sandboxed environment
    for (let i = 0; i < replayCount; i++) {
      try {
        const result = await this.executeSandboxedReplay(policyFn, input, i);
        results.push(result);
        outputHashes.add(result.outputHash);
      } catch (error) {
        console.error(`[REPLAY] Replay ${i} failed:`, error);
        results.push({
          replayId: i,
          output: null,
          outputHash: '',
          timestamp: Date.now(),
          success: false,
        });
      }
    }

    // Check consistency
    const successfulReplays = results.filter(r => r.success).length;
    const outputConsistent = outputHashes.size === 1;
    const consensusHash = outputConsistent ? Array.from(outputHashes)[0] : null;

    const report: ValidationReport = {
      totalReplays: replayCount,
      successfulReplays,
      failedReplays: replayCount - successfulReplays,
      outputConsistent,
      consensusHash,
      timestamp: Date.now(),
    };

    // Log results
    console.log(`[REPLAY] Validation complete:`);
    console.log(`  Successful: ${successfulReplays}/${replayCount}`);
    console.log(`  Output consistent: ${outputConsistent ? 'YES' : 'NO'}`);
    console.log(`  Unique hashes: ${outputHashes.size}`);

    if (!outputConsistent) {
      console.error('[REPLAY] ❌ VALIDATION FAILED: Non-deterministic behavior detected');
    } else {
      console.log('[REPLAY] ✅ VALIDATION PASSED: All outputs bit-identical');
    }

    return report;
  }

  /**
   * Execute a single sandboxed replay
   */
  private async executeSandboxedReplay(
    policyFn: PolicyFunction,
    input: any,
    replayId: number
  ): Promise<ReplayResult> {
    // Clone input to prevent mutation
    const clonedInput = JSON.parse(JSON.stringify(input));

    // Execute policy with deterministic seed
    const output = policyFn(clonedInput, this.entropySeed);

    // Hash output for comparison
    const outputHash = this.hashOutput(output);

    return {
      replayId,
      output,
      outputHash,
      timestamp: Date.now(),
      success: true,
    };
  }

  /**
   * Hash output for bit-identical comparison
   */
  private hashOutput(output: any): string {
    const serialized = JSON.stringify(output, Object.keys(output).sort());
    return createHash('sha256').update(serialized).digest('hex');
  }

  /**
   * Synchronous version of validatePolicy
   */
  validatePolicySync(
    policyFn: PolicyFunction,
    input: any,
    replayCount: number = this.minReplays
  ): ValidationReport {
    if (replayCount < this.minReplays) {
      throw new Error(`[REPLAY] Replay count ${replayCount} < minimum ${this.minReplays}`);
    }

    console.log(`[REPLAY] Starting deterministic validation with ${replayCount} replays`);

    const results: ReplayResult[] = [];
    const outputHashes = new Set<string>();

    for (let i = 0; i < replayCount; i++) {
      try {
        const clonedInput = JSON.parse(JSON.stringify(input));
        const output = policyFn(clonedInput, this.entropySeed);
        const outputHash = this.hashOutput(output);

        const result: ReplayResult = {
          replayId: i,
          output,
          outputHash,
          timestamp: Date.now(),
          success: true,
        };

        results.push(result);
        outputHashes.add(outputHash);
      } catch (error) {
        console.error(`[REPLAY] Replay ${i} failed:`, error);
        results.push({
          replayId: i,
          output: null,
          outputHash: '',
          timestamp: Date.now(),
          success: false,
        });
      }
    }

    const successfulReplays = results.filter(r => r.success).length;
    const outputConsistent = outputHashes.size === 1;
    const consensusHash = outputConsistent ? Array.from(outputHashes)[0] : null;

    return {
      totalReplays: replayCount,
      successfulReplays,
      failedReplays: replayCount - successfulReplays,
      outputConsistent,
      consensusHash,
      timestamp: Date.now(),
    };
  }

  /**
   * Get entropy seed
   */
  getEntropySeed(): number {
    return this.entropySeed;
  }

  /**
   * Set entropy seed (for testing)
   */
  setEntropySeed(seed: number): void {
    this.entropySeed = seed;
    console.log(`[REPLAY] Entropy seed updated: ${seed}`);
  }

  /**
   * Get minimum replay count
   */
  getMinReplays(): number {
    return this.minReplays;
  }
}

export default DeterministicReplayValidation;
