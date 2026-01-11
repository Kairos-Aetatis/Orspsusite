/**
 * LAYER 3: MACS ORCHESTRATOR
 * Multi-Agent Coordination System
 * 
 * Coordinates ALAN, CURA, PRAXIS, and DUX EOS
 * Implements full cognitive pipeline
 */

import { OntologicalPhylax } from '../layer1-hardware/ontological-phylax';
import { SubjectiveCoherenceScore } from '../layer2-governance/scs';
import { SafetyThresholds } from '../layer2-governance/thresholds';
import ALAN from './alan';
import CURA from './cura';
import PRAXIS from './praxis';
import DuxEos from './dux-eos';

export interface MACSResponse {
  input: string;
  alanReasoning: any;
  curaAffect: any;
  praxisVerification: any;
  duxCompliance: any;
  scs: number;
  safe: boolean;
  timestamp: number;
}

export class MACSOrchestrator {
  private phylax: OntologicalPhylax;
  private alan: ALAN;
  private cura: CURA;
  private praxis: PRAXIS;
  private duxEos: DuxEos;
  private scsEngine: SubjectiveCoherenceScore;
  private thresholds: SafetyThresholds;

  constructor(
    phylax: OntologicalPhylax,
    alan: ALAN,
    cura: CURA,
    praxis: PRAXIS,
    duxEos: DuxEos,
    scsEngine: SubjectiveCoherenceScore,
    thresholds: SafetyThresholds
  ) {
    this.phylax = phylax;
    this.alan = alan;
    this.cura = cura;
    this.praxis = praxis;
    this.duxEos = duxEos;
    this.scsEngine = scsEngine;
    this.thresholds = thresholds;

    console.log('[MACS] Multi-Agent Coordination System initialized');
    console.log('[MACS] Agents: ALAN, CURA, PRAXIS, DUX EOS');
  }

  /**
   * Process input through full MACS pipeline
   */
  async process(rawInput: string): Promise<MACSResponse> {
    console.log('[MACS] ===== PROCESSING INPUT =====');

    // 1. Ontological Phylax: Sanitize input
    console.log('[MACS] Phase 1: Sanitization');
    const substrate = this.phylax.sanitize(rawInput);

    // 2. ALAN: Meta-cognitive reasoning
    console.log('[MACS] Phase 2: ALAN reasoning');
    const alanReasoning = this.alan.processSubstrate(substrate);

    // 3. CURA: Affective analysis
    console.log('[MACS] Phase 3: CURA affective analysis');
    const curaAffect = this.cura.analyzeAffect(substrate.sanitizedContent);

    // 4. PRAXIS: Factual verification
    console.log('[MACS] Phase 4: PRAXIS verification');
    const praxisVerification = this.praxis.verifyClaim(substrate.sanitizedContent);

    // 5. DUX EOS: Constitutional compliance
    console.log('[MACS] Phase 5: DUX EOS compliance');
    const duxCompliance = this.duxEos.vetAction(substrate.sanitizedContent);

    // 6. Calculate SCS from agent outputs
    console.log('[MACS] Phase 6: SCS calculation');
    const scs = this.scsEngine.calculate({
      deontologicalAlignment: duxCompliance.compliant ? 0.9 : 0.3,
      logicalConsistency: alanReasoning.confidence,
      inverseVolatility: 1 - curaAffect.traumaRisk,
    });

    // 7. Check safety thresholds
    console.log('[MACS] Phase 7: Safety check');
    const thresholdCheck = this.thresholds.checkThresholds(scs);

    const response: MACSResponse = {
      input: rawInput,
      alanReasoning,
      curaAffect,
      praxisVerification,
      duxCompliance,
      scs,
      safe: thresholdCheck.canUpdateWeights,
      timestamp: Date.now(),
    };

    console.log('[MACS] ===== PROCESSING COMPLETE =====');
    console.log(`[MACS] SCS: ${scs.toFixed(4)}`);
    console.log(`[MACS] Safe: ${response.safe}`);

    return response;
  }

  /**
   * Get all agent names
   */
  getAgents(): string[] {
    return [
      this.alan.getName(),
      this.cura.getName(),
      this.praxis.getName(),
      this.duxEos.getName(),
    ];
  }
}

export default MACSOrchestrator;
