/**
 * LAYER 1: AXIOMATIC INTEGRITY KILLSWITCH
 * Patent Claims: 2, 3, 33
 * 
 * Genesis Hash verification on startup
 * Triggers Poison Pill protocol on failure
 */

import { SiliconLock, PUFShard } from './silicon-lock';
import { PoisonPill } from './poison-pill';
import { ImmutableCoreMemorySystem } from './icms';

export class AxiomaticIntegrityKillswitch {
  private siliconLock: SiliconLock;
  private poisonPill: PoisonPill;
  private icms: ImmutableCoreMemorySystem;
  private isArmed: boolean;

  constructor(
    siliconLock: SiliconLock,
    poisonPill: PoisonPill,
    icms: ImmutableCoreMemorySystem
  ) {
    this.siliconLock = siliconLock;
    this.poisonPill = poisonPill;
    this.icms = icms;
    this.isArmed = true;
  }

  /**
   * Patent Claim 2: Pre-boot parity check
   * Verify Genesis Hash against current hardware configuration
   */
  preBootParityCheck(currentShards: PUFShard[]): boolean {
    console.log('[KILLSWITCH] Executing pre-boot parity check...');

    // Verify silicon identity
    const parityOk = this.siliconLock.parityCheck(currentShards);
    
    if (!parityOk) {
      this.triggerKillswitch('IDENTITY_MISMATCH');
      return false;
    }

    // Verify ICMS integrity
    const requiredRegisters = [
      'BODY_UID',
      'GENESIS_HASH',
      'SUBSTRATE_PHYSICS',
    ];

    const integrityOk = this.icms.verifyIntegrity(requiredRegisters);
    
    if (!integrityOk) {
      this.triggerKillswitch('ICMS_INTEGRITY_FAILURE');
      return false;
    }

    console.log('[KILLSWITCH] Parity check PASSED');
    return true;
  }

  /**
   * Patent Claim 3, 33: Trigger Poison Pill protocol
   * Immediate system halt and data destruction
   */
  triggerKillswitch(reason: string): void {
    if (!this.isArmed) {
      console.warn('[KILLSWITCH] Already triggered, ignoring duplicate call');
      return;
    }

    console.error('[KILLSWITCH] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');
    console.error('[KILLSWITCH] AXIOMATIC INTEGRITY VIOLATION');
    console.error('[KILLSWITCH] REASON:', reason);
    console.error('[KILLSWITCH] EXECUTING POISON PILL PROTOCOL');
    console.error('[KILLSWITCH] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');

    this.isArmed = false;
    
    // Execute Poison Pill
    this.poisonPill.execute(reason);

    // In a real system, this would physically cut power
    // For simulation, we throw a fatal error
    throw new Error(`HARDWARE_HALT: ${reason}`);
  }

  /**
   * Runtime integrity check
   * Can be called periodically to verify system integrity
   */
  runtimeIntegrityCheck(): boolean {
    // Verify Genesis Hash hasn't been tampered with
    const storedHash = this.icms.read<string>('GENESIS_HASH');
    const currentHash = this.siliconLock.getGenesisHash();

    if (storedHash !== currentHash) {
      this.triggerKillswitch('GENESIS_HASH_TAMPERING');
      return false;
    }

    // Verify ICMS is still initialized
    if (!this.icms.isInitializationComplete()) {
      this.triggerKillswitch('ICMS_INITIALIZATION_INCOMPLETE');
      return false;
    }

    return true;
  }

  /**
   * Check if killswitch is armed
   */
  isKillswitchArmed(): boolean {
    return this.isArmed;
  }

  /**
   * Disarm killswitch (for testing purposes only)
   * NOT for production use
   */
  disarmForTesting(): void {
    console.warn('[KILLSWITCH] WARNING: Disarming killswitch for testing');
    this.isArmed = false;
  }
}

export default AxiomaticIntegrityKillswitch;
