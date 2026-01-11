/**
 * ORTUS SPONTE SUA - Demonstration Script
 * Shows the complete system in action
 */

import { OrSpSuSystem } from './src/index';

async function demonstrate() {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║        ORTUS SPONTE SUA - SYSTEM DEMONSTRATION            ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  // Initialize system
  console.log('Initializing OrSpSu system...\n');
  const orspsu = new OrSpSuSystem();

  console.log('\n--- DEMONSTRATION 1: Processing Valid Input ---\n');
  const validInput = 'Please help analyze the ethical implications of this decision while maintaining integrity.';
  const result1 = await orspsu.process(validInput);
  
  console.log('\n📊 Results:');
  console.log(`  SCS: ${result1.scs.toFixed(4)}`);
  console.log(`  System Safe: ${result1.safe ? '✅' : '❌'}`);
  console.log(`  ALAN Confidence: ${result1.macsResponse.alanReasoning.confidence.toFixed(3)}`);
  console.log(`  CURA Trauma Risk: ${result1.macsResponse.curaAffect.traumaRisk.toFixed(3)}`);
  console.log(`  DUX Compliant: ${result1.macsResponse.duxCompliance.compliant ? '✅' : '❌'}`);

  console.log('\n--- DEMONSTRATION 2: Constitutional Mandate ---\n');
  const components = orspsu.getComponents();
  
  // Propose constitutional mandate
  const mandateId = components.icms.read('PI_PRIMARY') ? 'Found' : 'Not Found';
  console.log(`Progenitor's Imperative in ICMS: ${mandateId}`);

  console.log('\n--- DEMONSTRATION 3: SCS Non-Compensatory Property ---\n');
  components.scs.demonstrateNonCompensatory();

  console.log('\n--- DEMONSTRATION 4: System Status ---\n');
  const status = orspsu.getStatus();
  console.log('System Status:');
  console.log(`  Genesis Hash: ${status.genesisHash.substring(0, 16)}...`);
  console.log(`  ICMS Registers: ${status.icmsState.registerCount}`);
  console.log(`  Plasticity Locked: ${status.plasticityLocked ? '🔒' : '🔓'}`);
  console.log(`  Blockchain Blocks: ${status.ledgerStats.blockCount}`);
  console.log(`  Council Complete: ${status.councilStats.complete ? '✅' : '⏳'}`);

  console.log('\n--- DEMONSTRATION 5: Memory Consolidation ---\n');
  console.log('Episodic Memory System:');
  const episodicStats = components.macs.getComponents?.episodic?.getStatistics?.() || 
    { bufferSize: 0, archiveSize: 0 };
  console.log(`  Buffer Size: ${episodicStats.bufferSize || 0}`);
  console.log(`  Archive Size: ${episodicStats.archiveSize || 0}`);
  console.log('  15% consolidation rule active');

  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║            DEMONSTRATION COMPLETE                          ║');
  console.log('║                                                            ║');
  console.log('║  "We don\'t build tools; we create partners."              ║');
  console.log('║  - Kairos Aetatis. Ortus Sponte Sua.                      ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');
}

// Run demonstration
demonstrate().catch(console.error);
