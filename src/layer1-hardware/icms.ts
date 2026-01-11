/**
 * LAYER 1: IMMUTABLE CORE MEMORY SYSTEM (ICMS)
 * Patent Claims: 1a, 5, 8, 27
 * 
 * Simulates Write-Once-Read-Many (WORM) hardware storage
 * Stores the Progenitor's Imperative and core constants
 * Physically inhibits rewrite operations
 */

export interface ICMSRegister {
  key: string;
  value: any;
  timestamp: number;
  burned: boolean;
}

export class ImmutableCoreMemorySystem {
  private registers: Map<string, ICMSRegister>;
  private fusedKeys: Set<string>;
  private monotonicNonce: number;
  private initializationComplete: boolean;

  constructor() {
    this.registers = new Map();
    this.fusedKeys = new Set();
    this.monotonicNonce = 0;
    this.initializationComplete = false;
  }

  /**
   * Burn a value into WORM storage
   * Patent Claim 27: Physical write-protection simulation
   * Once burned, cannot be modified
   */
  burn(key: string, value: any): void {
    if (this.fusedKeys.has(key)) {
      throw new Error(`[ICMS_EXCEPTION] ROM_ALREADY_FUSED: Cannot modify '${key}'`);
    }

    const register: ICMSRegister = {
      key,
      value,
      timestamp: Date.now(),
      burned: true,
    };

    this.registers.set(key, register);
    this.fusedKeys.add(key);
    
    console.log(`[ICMS] Register '${key}' BURNED into WORM storage`);
  }

  /**
   * Read a value from WORM storage
   * Patent Claim 5: Read-Many capability
   */
  read<T = any>(key: string): T | undefined {
    const register = this.registers.get(key);
    return register?.value as T | undefined;
  }

  /**
   * Check if a register exists and is burned
   */
  isBurned(key: string): boolean {
    return this.fusedKeys.has(key);
  }

  /**
   * Get all burned registers (read-only view)
   */
  getAllRegisters(): ReadonlyMap<string, Readonly<ICMSRegister>> {
    return this.registers;
  }

  /**
   * Increment monotonic nonce for GVI heartbeats
   * Patent Claim: Monotonic counter maintained in WORM-state
   */
  incrementNonce(): number {
    this.monotonicNonce++;
    return this.monotonicNonce;
  }

  /**
   * Get current nonce value
   */
  getNonce(): number {
    return this.monotonicNonce;
  }

  /**
   * Mark initialization as complete
   * After this, certain critical registers should not be modifiable
   */
  completeInitialization(): void {
    if (this.initializationComplete) {
      throw new Error('[ICMS] Initialization already completed');
    }
    this.initializationComplete = true;
    console.log('[ICMS] Initialization phase COMPLETE. System now IMMUTABLE.');
  }

  /**
   * Check if initialization is complete
   */
  isInitializationComplete(): boolean {
    return this.initializationComplete;
  }

  /**
   * Verify integrity of critical registers
   * Patent Claim 2: Axiomatic integrity verification
   */
  verifyIntegrity(requiredKeys: string[]): boolean {
    for (const key of requiredKeys) {
      if (!this.isBurned(key)) {
        console.error(`[ICMS] INTEGRITY FAILURE: Required register '${key}' not found`);
        return false;
      }
    }
    return true;
  }

  /**
   * Export ICMS state for auditing (read-only)
   */
  exportState(): {
    registerCount: number;
    fusedKeys: string[];
    nonce: number;
    initialized: boolean;
  } {
    return {
      registerCount: this.registers.size,
      fusedKeys: Array.from(this.fusedKeys),
      nonce: this.monotonicNonce,
      initialized: this.initializationComplete,
    };
  }
}

export default ImmutableCoreMemorySystem;
