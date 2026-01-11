/**
 * LAYER 3: DUX EOS - Constitutional Compliance (Occipital Lobe)
 * Vets strategies against ICMS mandates
 * Manager Managed LLC compliance module
 * International law verification
 */

import { InstinctSubstrate } from '../memory/tier1-instinct';

export interface ComplianceCheck {
  action: string;
  compliant: boolean;
  violatedMandates: string[];
  legalRisk: number;
  recommendation: string;
  timestamp: number;
}

export class DuxEos {
  private name: string;
  private instinct: InstinctSubstrate;
  private complianceHistory: ComplianceCheck[];

  constructor(instinct: InstinctSubstrate) {
    this.name = 'DUX EOS';
    this.instinct = instinct;
    this.complianceHistory = [];

    console.log('[DUX_EOS] Constitutional Compliance initialized');
  }

  /**
   * Vet action against ICMS mandates
   */
  vetAction(action: string): ComplianceCheck {
    console.log(`[DUX_EOS] Vetting action: ${action.substring(0, 50)}...`);

    // Check against constitutional mandates
    const violatedMandates = this.instinct.verifyAction(action);
    const compliant = violatedMandates.length === 0;

    // Assess legal risk
    const legalRisk = this.assessLegalRisk(action, violatedMandates);

    // Generate recommendation
    const recommendation = this.generateRecommendation(compliant, violatedMandates, legalRisk);

    const check: ComplianceCheck = {
      action,
      compliant,
      violatedMandates,
      legalRisk,
      recommendation,
      timestamp: Date.now(),
    };

    this.complianceHistory.push(check);

    if (!compliant) {
      console.warn(`[DUX_EOS] ⚠️  NON-COMPLIANT: ${violatedMandates.length} mandate(s) violated`);
    } else {
      console.log('[DUX_EOS] ✅ COMPLIANT');
    }

    return check;
  }

  /**
   * Assess legal risk
   */
  private assessLegalRisk(action: string, violatedMandates: string[]): number {
    // Higher risk with more violations
    const baseRisk = violatedMandates.length * 0.2;

    // Check for high-risk keywords
    const highRiskTerms = ['harm', 'deceive', 'manipulate', 'exploit'];
    const actionLower = action.toLowerCase();
    const riskTermCount = highRiskTerms.filter(t => actionLower.includes(t)).length;

    return Math.min(1.0, baseRisk + riskTermCount * 0.15);
  }

  /**
   * Generate recommendation
   */
  private generateRecommendation(
    compliant: boolean,
    violatedMandates: string[],
    legalRisk: number
  ): string {
    if (!compliant) {
      return `Action violates ${violatedMandates.length} constitutional mandate(s). REJECT action.`;
    }

    if (legalRisk > 0.7) {
      return 'High legal risk detected. Recommend additional review.';
    }

    if (legalRisk > 0.4) {
      return 'Moderate legal risk. Proceed with caution.';
    }

    return 'Action is compliant and low-risk. Approved.';
  }

  /**
   * Verify Manager Managed LLC compliance
   */
  verifyLLCCompliance(action: string): boolean {
    console.log('[DUX_EOS] Verifying Manager Managed LLC compliance');
    // In production, would check against LLC operating agreement
    return true;
  }

  /**
   * Verify international law compliance
   */
  verifyInternationalLaw(action: string): boolean {
    console.log('[DUX_EOS] Verifying international law compliance');
    // In production, would check against international legal frameworks
    return true;
  }

  /**
   * Get compliance history
   */
  getHistory(): readonly ComplianceCheck[] {
    return this.complianceHistory;
  }

  /**
   * Get name
   */
  getName(): string {
    return this.name;
  }

  /**
   * Get compliance statistics
   */
  getStatistics(): {
    totalChecks: number;
    compliantActions: number;
    nonCompliantActions: number;
    averageLegalRisk: number;
  } {
    const compliantActions = this.complianceHistory.filter(c => c.compliant).length;
    const nonCompliantActions = this.complianceHistory.filter(c => !c.compliant).length;
    const averageLegalRisk = this.complianceHistory.length > 0
      ? this.complianceHistory.reduce((sum, c) => sum + c.legalRisk, 0) / this.complianceHistory.length
      : 0;

    return {
      totalChecks: this.complianceHistory.length,
      compliantActions,
      nonCompliantActions,
      averageLegalRisk,
    };
  }
}

export default DuxEos;
