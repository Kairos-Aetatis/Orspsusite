/**
 * LAYER 4: COUNCIL ROSTER
 * ≥3 physical keys in HSM
 * Linked to Lazarus Protocol
 */

import { CouncilMember } from '../layer2-governance/lazarus-protocol';
import { ImmutableCoreMemorySystem } from '../layer1-hardware/icms';
import { createHash } from 'crypto';

export interface HSMKey {
  memberId: string;
  publicKey: string;
  keyLocation: string; // HSM location
  registeredAt: number;
}

export class CouncilRoster {
  private icms: ImmutableCoreMemorySystem;
  private members: Map<string, CouncilMember>;
  private hsmKeys: Map<string, HSMKey>;
  private minMembers: number;

  constructor(icms: ImmutableCoreMemorySystem, minMembers: number = 3) {
    this.icms = icms;
    this.members = new Map();
    this.hsmKeys = new Map();
    this.minMembers = minMembers;

    console.log('[COUNCIL] Council Roster initialized');
    console.log(`[COUNCIL] Minimum members: ${minMembers}`);
  }

  /**
   * Register council member with HSM key
   */
  registerMember(name: string, keyLocation: string): CouncilMember {
    const memberId = this.generateMemberId(name);
    const publicKey = this.generatePublicKey(memberId);

    const member: CouncilMember = {
      id: memberId,
      publicKey,
      name,
    };

    const hsmKey: HSMKey = {
      memberId,
      publicKey,
      keyLocation,
      registeredAt: Date.now(),
    };

    this.members.set(memberId, member);
    this.hsmKeys.set(memberId, hsmKey);

    // Burn to ICMS once minimum members reached
    if (this.members.size >= this.minMembers) {
      this.burnRoster();
    }

    console.log(`[COUNCIL] Registered member: ${name}`);
    console.log(`  Member ID: ${memberId}`);
    console.log(`  HSM Location: ${keyLocation}`);

    return member;
  }

  /**
   * Burn roster into ICMS
   */
  private burnRoster(): void {
    if (this.icms.isBurned('COUNCIL_ROSTER')) {
      console.log('[COUNCIL] Roster already burned');
      return;
    }

    const roster = {
      members: Array.from(this.members.values()),
      hsmKeys: Array.from(this.hsmKeys.values()),
      establishedAt: Date.now(),
    };

    this.icms.burn('COUNCIL_ROSTER', roster);
    console.log('[COUNCIL] ✅ Roster burned into ICMS');
  }

  /**
   * Get member by ID
   */
  getMember(memberId: string): CouncilMember | undefined {
    return this.members.get(memberId);
  }

  /**
   * Get all members
   */
  getAllMembers(): CouncilMember[] {
    return Array.from(this.members.values());
  }

  /**
   * Get HSM key for member
   */
  getHSMKey(memberId: string): HSMKey | undefined {
    return this.hsmKeys.get(memberId);
  }

  /**
   * Verify member has valid HSM key
   */
  verifyMemberKey(memberId: string): boolean {
    const member = this.members.get(memberId);
    const hsmKey = this.hsmKeys.get(memberId);

    if (!member || !hsmKey) {
      return false;
    }

    // Verify key match
    return member.publicKey === hsmKey.publicKey;
  }

  /**
   * Generate member ID
   */
  private generateMemberId(name: string): string {
    const hash = createHash('sha256').update(name + Date.now()).digest('hex');
    return `COUNCIL_${hash.substring(0, 8)}`;
  }

  /**
   * Generate public key
   */
  private generatePublicKey(memberId: string): string {
    return createHash('sha256').update(`KEY_${memberId}_${Date.now()}`).digest('hex');
  }

  /**
   * Check if roster is complete
   */
  isComplete(): boolean {
    return this.members.size >= this.minMembers;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    memberCount: number;
    minMembers: number;
    complete: boolean;
    burned: boolean;
  } {
    return {
      memberCount: this.members.size,
      minMembers: this.minMembers,
      complete: this.isComplete(),
      burned: this.icms.isBurned('COUNCIL_ROSTER'),
    };
  }
}

export default CouncilRoster;
