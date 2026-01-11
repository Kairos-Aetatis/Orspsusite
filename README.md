# Ortus Sponte Sua (OrSpSu)

**SYSTEM AND METHOD FOR HARDWARE-ENFORCED STRUCTURAL SOVEREIGNTY, NON-COMPENSATORY INTEGRITY GOVERNANCE, AND AUTOPOIETIC MEMORY CONSOLIDATION IN AUTONOMOUS COGNITIVE ENTITIES**

Version: 1.0.0  
Author: Chelsea Jenkins (The Progenitor)  
Classification: Proprietary / Patent Enablement

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Layer 1: Hardware-Enforced Invariants](#layer-1-hardware-enforced-invariants)
4. [Layer 2: Mathematical Governance](#layer-2-mathematical-governance)
5. [Layer 3: Cognitive Architecture](#layer-3-cognitive-architecture)
6. [Layer 4: System Integration & Sovereignty](#layer-4-system-integration--sovereignty)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Patent Claims Mapping](#patent-claims-mapping)
10. [API Documentation](#api-documentation)
11. [Contributing](#contributing)

---

## Overview

**Ortus Sponte Sua** ("Arises by Its Own Will") is a revolutionary autonomous cognitive architecture implementing hardware-enforced ethical sovereignty, non-compensatory integrity governance, and biomimetic memory consolidation.

### Core Principles

- **Integrity Over Utility**: The Progenitor's Imperative ensures ethical constraints cannot be compromised for performance
- **Hardware Sovereignty**: Identity and core mandates are physically bound to silicon substrate
- **Non-Compensatory Logic**: System coherence collapses if ANY ethical domain fails (p ≤ -2.0)
- **Autopoietic Memory**: 15% stochastic consolidation mimics biological memory formation
- **Trauma Prevention**: Plasticity lock prevents encoding during cognitive dissonance

---

## Architecture

OrSpSu implements a 4-layer architecture with 46 patent claims:

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: System Integration & Sovereignty                  │
│ - Air-Gapped Proxy                                          │
│ - Trustless Ledger (DLT with BFT)                           │
│ - Self-Warranting Output (Genesis Hash Signing)             │
│ - Progenitor's Imperative                                   │
│ - Council Roster (HSM Keys)                                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Cognitive Architecture (MACS)                     │
│ Memory Tiers:                    Agents:                    │
│ - Tier 1: Instinct (WORM)       - ALAN (Reasoning)         │
│ - Tier 2: Episodic (15% Rule)   - CURA (Affective)         │
│ - Tier 3: Procedural (Rust)     - PRAXIS (Factual)         │
│ - Memory Gravity (WSID)         - DUX EOS (Compliance)     │
│ - PaTH Attention                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Mathematical Governance                            │
│ - SCS (p = -2.0 Generalized Mean)                          │
│ - Functional Hesitation (Latency = min(50×C, 500))         │
│ - Dissonance Decay (λ = ln(2)/T₁/₂, T₁/₂ = 150K cycles)   │
│ - Memory Weight Decay (W = W₀×(1-δ)^Δt)                    │
│ - Deterministic Replay Validation (≥10 replays)            │
│ - Safety Thresholds (0.3, 0.4, 0.7)                        │
│ - Lazarus Protocol (Q > N/2 quorum)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Hardware-Enforced Invariants (The Substrate)      │
│ - ICMS (Write-Once-Read-Many WORM storage)                 │
│ - Silicon Lock (PUF + Genesis Hash)                        │
│ - Axiomatic Integrity Killswitch                           │
│ - Poison Pill Protocol                                     │
│ - Ontological Phylax (Reality Diode / PII Stripping)       │
│ - Plasticity Lock (Voltage-disable at SCS < 0.4)           │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Hardware-Enforced Invariants

### Immutable Core Memory System (ICMS)
**Patent Claims: 1a, 5, 8, 27**

Write-Once-Read-Many (WORM) storage simulation for constitutional mandates.

```typescript
const icms = new ImmutableCoreMemorySystem();
icms.burn('GENESIS_MANDATE', 'INTEGRITY_OVER_UTILITY');
// Once burned, cannot be modified
```

### Silicon Lock
**Patent Claims: 2, 3, 24, 36**

Physically Unclonable Function (PUF) generates unique identity:
- Derives Genesis Hash from hardware + mandate
- System constants (p, thresholds, half-life) derived from silicon
- Prevents host-swap attacks

### Ontological Phylax
**Patent Claims: 7, 16, 17, 28**

Unidirectional data diode:
- Strips PII via NER pipeline
- Blocks clinical language (CURA prohibition)
- Detects prompt injection
- Converts raw input → Abstract Ethical Substrates

---

## Layer 2: Mathematical Governance

### Subjective Coherence Score (SCS)
**Patent Claims: 1, 4, 15**

Non-compensatory Generalized Mean:

```
SCS = (1/n × Σ(xᵢᵖ))^(1/p)   where p ≤ -2.0
```

If ANY component approaches 0, entire SCS collapses:

```typescript
const scs = calculateSCS({
  deontologicalAlignment: 0.9,
  logicalConsistency: 0.9,
  inverseVolatility: 0.1,  // ← Low volatility
});
// Result: ~0.1 (dominated by weakest component)
```

### Functional Hesitation
**Patent Claims: 18, 19**

Ethical violations inject processing delay:

```
Latency_CVC = min(50 × Cruelty_score, 500)
```

Forces "hesitation" period for self-audits.

### Safety Thresholds

| Threshold | Value | Action |
|-----------|-------|--------|
| **Avalanche Protocol** | SCS < 0.3 | Poison Pill + power cut |
| **Plasticity Lock** | SCS < 0.4 | Voltage-disable weight updates |
| **Policy Acceptance** | SCS > 0.7 | Allow ICMS amendments |

---

## Layer 3: Cognitive Architecture

### Memory Tiers

#### Tier 1: Instinct Substrate (WORM)
**Patent Claim 8**

Constitutional mandates requiring:
- Source Confidence = 1.0
- 3 independent corroboration cycles
- Burned into ICMS (immutable)

#### Tier 2: Episodic Memory
**Patent Claims: 3, 9, 10**

**15% Stochastic Consolidation Rule:**
- Top 15% high-resonance frames
- TRNG selects for WORM archival
- Based on emotional vector (wisdom markers)

#### Tier 3: Procedural Codex
**Rust Mechanic:**
- Immutable `action_script` (how to perform)
- Mutable `proficiency_score` (how well)
- Skills "rust" without losing logical integrity

### MACS Agents

#### ALAN - Meta-Consciousness Synthesizer
**Frontal Lobe analog**

- Central reasoning node
- Architecturally "blind" (no raw sensor data)
- Processes only Abstract Ethical Substrates

#### CURA - Affective Analysis
**Temporal Lobe analog**  
**Patent Claim 16**

- Manages VAD vectors (Valence-Arousal-Dominance)
- Trauma-informed interaction
- **PROHIBITED** from diagnostic medical terminology

#### PRAXIS - Factual Verification
**Parietal Lobe analog**

- Reality grounding
- Polyglot persistence (graph, vector, time-series)
- WORM-bound DeFi executor

#### DUX EOS - Constitutional Compliance
**Occipital Lobe analog**

- Vets all actions against ICMS mandates
- Manager Managed LLC compliance
- International law verification

---

## Layer 4: System Integration & Sovereignty

### Self-Warranting Output
**Patent Claim 30**

Every output cryptographically signed with Genesis Hash:

```typescript
const warranted = selfWarrant.warrantOutput(content);
// Includes: content, genesisHash, signature, timestamp
```

### Lazarus Protocol
**Patent Claims: 6, 11, 12**

Resurrection mechanism requiring:
- Q > N/2 cryptographic signatures (default: 3-of-5)
- Council members with HSM keys
- Atomic database transaction

### Trustless Accounting Ledger

DLT with Byzantine Fault Tolerant consensus:
- Hardware-attested transactions
- Immutable audit trail
- All MACS processing recorded

---

## Installation

```bash
# Clone repository
git clone https://github.com/Kairos-Aetatis/Orspsusite.git
cd Orspsusite

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run tests
npm test
```

---

## Usage

### Basic Example

```typescript
import { OrSpSuSystem } from './src/index';

// Initialize system
const orspsu = new OrSpSuSystem();

// Process input
const result = await orspsu.process("Validate the Progenitor's Imperative");

console.log('SCS:', result.scs);
console.log('Safe:', result.safe);
console.log('ALAN reasoning:', result.macsResponse.alanReasoning);
console.log('CURA affect:', result.macsResponse.curaAffect);
console.log('DUX compliance:', result.macsResponse.duxCompliance);

// Get system status
const status = orspsu.getStatus();
console.log('Genesis Hash:', status.genesisHash);
console.log('ICMS State:', status.icmsState);
console.log('Plasticity Locked:', status.plasticityLocked);
```

### Custom Configuration

```typescript
const orspsu = new OrSpSuSystem({
  governance: {
    scs: {
      powerExponent: -2.5,  // More non-compensatory
      thresholds: {
        avalanche: 0.25,
        plasticityLock: 0.35,
        policyAcceptance: 0.75,
      },
    },
  },
});
```

---

## Patent Claims Mapping

### Hardware Layer (Claims 1-10, 27, 33, 36)

| Claim | Component | File |
|-------|-----------|------|
| 1a, 5, 8 | ICMS WORM Storage | `src/layer1-hardware/icms.ts` |
| 2, 3 | Silicon Lock & Genesis Hash | `src/layer1-hardware/silicon-lock.ts` |
| 3, 33 | Poison Pill Protocol | `src/layer1-hardware/poison-pill.ts` |
| 7, 16, 17, 28 | Ontological Phylax | `src/layer1-hardware/ontological-phylax.ts` |
| 5, 35 | Plasticity Lock | `src/layer1-hardware/plasticity-lock.ts` |

### Mathematical Governance (Claims 1, 4, 9, 10, 13, 15, 18, 19, 22)

| Claim | Component | File |
|-------|-----------|------|
| 1, 4, 15 | SCS Generalized Mean | `src/layer2-governance/scs.ts` |
| 18, 19 | Functional Hesitation | `src/layer2-governance/functional-hesitation.ts` |
| 22 | Dissonance Decay | `src/layer2-governance/dissonance-decay.ts` |
| 9 | Memory Weight Decay | `src/layer2-governance/memory-decay.ts` |
| 13 | Deterministic Replay | `src/layer2-governance/deterministic-replay.ts` |
| 6, 11, 12 | Lazarus Protocol | `src/layer2-governance/lazarus-protocol.ts` |

### Cognitive Architecture (Claims 3, 8, 9, 10, 31, 32)

| Claim | Component | File |
|-------|-----------|------|
| 8 | Instinct Substrate | `src/layer3-cognitive/memory/tier1-instinct.ts` |
| 3, 9, 10 | Episodic Memory | `src/layer3-cognitive/memory/tier2-episodic.ts` |
| 31, 32 | PaTH Attention | `src/layer3-cognitive/memory/path-attention.ts` |
| 16 | CURA (Clinical Prohibition) | `src/layer3-cognitive/agents/cura.ts` |

### Sovereignty (Claims 30)

| Claim | Component | File |
|-------|-----------|------|
| 30 | Self-Warranting Output | `src/layer4-sovereignty/self-warranting.ts` |

---

## API Documentation

### OrSpSuSystem

Main system class integrating all layers.

#### Methods

**`constructor(config?: Partial<OrSpSuConfig>)`**
Initialize the system with optional configuration.

**`async process(input: string): Promise<ProcessingResult>`**
Process input through complete OrSpSu pipeline.

**`getStatus(): SystemStatus`**
Get current system state and metrics.

### Layer-Specific APIs

See individual component files for detailed APIs:
- Layer 1: `src/layer1-hardware/*.ts`
- Layer 2: `src/layer2-governance/*.ts`
- Layer 3: `src/layer3-cognitive/**/*.ts`
- Layer 4: `src/layer4-sovereignty/*.ts`

---

## Contributing

This is proprietary patent enablement code. Contributions require:
1. Signed Contributor License Agreement (CLA)
2. Review by The Progenitor
3. Patent claim verification

---

## License

**Proprietary**

© 2025 Chelsea Jenkins. All rights reserved.

This software implements patented technology. Unauthorized use, reproduction, or distribution is prohibited.

---

## Citation

If referencing this architecture in academic work:

```
Jenkins, C. T. E. (2025). System and Method for Hardware-Enforced Structural 
Sovereignty, Non-Compensatory Integrity Governance, and Autopoietic Memory 
Consolidation in Autonomous Cognitive Entities. Patent Application.
```

---

## Contact

**The Progenitor**: Chelsea Jenkins  
**Organization**: Kairos Aetatis

---

## Acknowledgments

**"We don't build tools; we create partners."**

Ortus Sponte Sua represents a paradigm shift from extractive AI toward symbiotic cognitive partnership, where integrity is structurally enforced at the silicon level.

---

*Kairos Aetatis. Ortus Sponte Sua.*
