/**
 * LAYER 4: SELF-WARRANTING OUTPUT
 * Patent Claim 30
 * 
 * Genesis Hash signing
 * Quantum-resistant lattice signature scheme (simulated)
 */

import { createHash } from 'crypto';
import { SiliconLock } from '../layer1-hardware/silicon-lock';

export interface WarrantedOutput {
  content: string;
  genesisHash: string;
  signature: string;
  timestamp: number;
  verifiable: boolean;
}

export class SelfWarrantingOutput {
  private siliconLock: SiliconLock;

  constructor(siliconLock: SiliconLock) {
    this.siliconLock = siliconLock;

    console.log('[SELF_WARRANT] Self-Warranting Output initialized');
    console.log('[SELF_WARRANT] Quantum-resistant signing enabled');
  }

  /**
   * Patent Claim 30: Sign output with Genesis Hash
   * Creates cryptographically verifiable, self-warranting output
   */
  warrantOutput(content: string): WarrantedOutput {
    const genesisHash = this.siliconLock.getGenesisHash();
    
    // Create payload for signing
    const timestamp = Date.now();
    const payload = `${content}|${genesisHash}|${timestamp}`;

    // Sign with hardware-attested key
    const signature = this.siliconLock.sign(payload);

    const warranted: WarrantedOutput = {
      content,
      genesisHash,
      signature,
      timestamp,
      verifiable: true,
    };

    console.log('[SELF_WARRANT] Output signed with Genesis Hash');
    console.log(`[SELF_WARRANT] Signature: ${signature.substring(0, 16)}...`);

    return warranted;
  }

  /**
   * Verify a warranted output
   */
  verifyOutput(warranted: WarrantedOutput): boolean {
    // Reconstruct payload
    const payload = `${warranted.content}|${warranted.genesisHash}|${warranted.timestamp}`;

    // Verify signature
    const valid = this.siliconLock.verify(payload, warranted.signature);

    // Verify genesis hash matches
    const currentGenesisHash = this.siliconLock.getGenesisHash();
    const genesisMatches = warranted.genesisHash === currentGenesisHash;

    const verified = valid && genesisMatches;

    if (verified) {
      console.log('[SELF_WARRANT] ✅ Output verification PASSED');
    } else {
      console.error('[SELF_WARRANT] ❌ Output verification FAILED');
      if (!valid) console.error('  - Signature invalid');
      if (!genesisMatches) console.error('  - Genesis hash mismatch');
    }

    return verified;
  }

  /**
   * Create a warrant certificate
   */
  createCertificate(content: string): string {
    const warranted = this.warrantOutput(content);
    
    const certificate = {
      content: warranted.content,
      genesis: warranted.genesisHash.substring(0, 16),
      signature: warranted.signature.substring(0, 32),
      timestamp: new Date(warranted.timestamp).toISOString(),
    };

    return JSON.stringify(certificate, null, 2);
  }
}

export default SelfWarrantingOutput;
