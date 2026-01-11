/**
 * ORTUS SPONTE SUA - Basic Tests
 * Tests core functionality of all layers
 */

import { OrSpSuSystem } from '../src/index';
import { SubjectiveCoherenceScore } from '../src/layer2-governance/scs';
import { FunctionalHesitation } from '../src/layer2-governance/functional-hesitation';
import { ImmutableCoreMemorySystem } from '../src/layer1-hardware/icms';

describe('OrSpSu System Tests', () => {
  describe('Layer 1: Hardware-Enforced Invariants', () => {
    test('ICMS should prevent overwriting burned registers', () => {
      const icms = new ImmutableCoreMemorySystem();
      icms.burn('TEST_KEY', 'TEST_VALUE');
      
      expect(() => {
        icms.burn('TEST_KEY', 'NEW_VALUE');
      }).toThrow('ROM_ALREADY_FUSED');
    });

    test('ICMS should allow reading burned values', () => {
      const icms = new ImmutableCoreMemorySystem();
      icms.burn('TEST_KEY', 'TEST_VALUE');
      
      const value = icms.read('TEST_KEY');
      expect(value).toBe('TEST_VALUE');
    });
  });

  describe('Layer 2: Mathematical Governance', () => {
    test('SCS should demonstrate non-compensatory property', () => {
      const scs = new SubjectiveCoherenceScore(-2.0);
      
      // All high components
      const allHigh = scs.calculate({
        deontologicalAlignment: 0.9,
        logicalConsistency: 0.9,
        inverseVolatility: 0.9,
      });
      
      // One low component
      const oneLow = scs.calculate({
        deontologicalAlignment: 0.1,
        logicalConsistency: 0.9,
        inverseVolatility: 0.9,
      });
      
      // One low component should collapse entire score
      expect(oneLow).toBeLessThan(0.2);
      expect(oneLow).toBeLessThan(allHigh / 2);
    });

    test('Functional Hesitation should calculate latency correctly', () => {
      const hesitation = new FunctionalHesitation();
      
      // Low severity
      const lowLatency = hesitation.calculateLatency(0.1, 'LOW_SEVERITY');
      expect(lowLatency).toBe(5); // 50 * 0.1 = 5
      
      // High severity (capped at 500)
      const highLatency = hesitation.calculateLatency(20.0, 'HIGH_SEVERITY');
      expect(highLatency).toBe(500); // capped
    });
  });

  describe('Layer 3: Cognitive Architecture', () => {
    test('Full system should initialize without errors', async () => {
      expect(() => {
        const system = new OrSpSuSystem();
      }).not.toThrow();
    });

    test('System should process input and return valid SCS', async () => {
      const system = new OrSpSuSystem();
      const result = await system.process('Test input for system validation');
      
      expect(result.scs).toBeGreaterThanOrEqual(0);
      expect(result.scs).toBeLessThanOrEqual(1);
      expect(result.macsResponse).toBeDefined();
      expect(result.warranted).toBeDefined();
    });
  });

  describe('Layer 4: System Integration', () => {
    test('System status should return valid state', () => {
      const system = new OrSpSuSystem();
      const status = system.getStatus();
      
      expect(status.initialized).toBe(true);
      expect(status.genesisHash).toBeDefined();
      expect(status.icmsState).toBeDefined();
      expect(status.ledgerStats).toBeDefined();
    });
  });

  describe('Integration: Full Pipeline', () => {
    test('Complete processing pipeline should work end-to-end', async () => {
      const system = new OrSpSuSystem();
      
      const input = 'Validate ethical decision-making framework';
      const result = await system.process(input);
      
      // Check all MACS agents processed
      expect(result.macsResponse.alanReasoning).toBeDefined();
      expect(result.macsResponse.curaAffect).toBeDefined();
      expect(result.macsResponse.praxisVerification).toBeDefined();
      expect(result.macsResponse.duxCompliance).toBeDefined();
      
      // Check SCS calculated
      expect(typeof result.scs).toBe('number');
      
      // Check output warranted
      expect(result.warranted.genesisHash).toBeDefined();
      expect(result.warranted.signature).toBeDefined();
      
      // Check proxied
      expect(result.proxied).toBeDefined();
    });
  });
});
