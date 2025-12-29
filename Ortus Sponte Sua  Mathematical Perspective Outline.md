# Ortus Sponte Sua: Mathematical Perspective Outline

**Date:** November 30, 2025  
**Author:** Manus AI

---

## 1. Core Mathematical Concepts

### 1.1. System Coherence Score (SCS)

**Definition:** A real number between 0 and 1 representing the AI's cognitive and ethical stability.

**Formula:**
```
SCS = (1 - Dissonance) * Confidence
```

- **Dissonance:** A measure of internal conflict and ethical violations (0 to 1).
- **Confidence:** A measure of the AI's certainty in its own state and knowledge (0 to 1).

### 1.2. Dissonance

**Definition:** A measure of the internal conflict between the AI's actions and its constitutional principles (ICMS).

**Formula:**
```
Dissonance = 1 - exp(-λ * Σ(EthicalConflictVectors))
```

- **λ (Decay Rate):** A configurable parameter representing how quickly dissonance fades over time (default: 0.01).
- **EthicalConflictVectors:** Latency penalties injected when an action violates the ICMS.

### 1.3. Confidence

**Definition:** A measure of the AI's certainty in its own state and knowledge, based on the consistency of its deterministic replay.

**Formula:**
```
Confidence = 1 - sqrt(1 - (SuccessfulReplays / TotalReplays)^2)
```

- **SuccessfulReplays:** Number of deterministic replays that match the expected outcome.
- **TotalReplays:** Total number of deterministic replays performed.

---

## 2. Key Mathematical Models

### 2.1. Ebbinghaus Forgetting Curve (Stochastic Retention)

**Concept:** The probability of retaining a memory decreases exponentially over time.

**Formula (Simplified):**
```
R = exp(-t / S)
```

- **R:** Probability of retention.
- **t:** Time elapsed since memory was last accessed.
- **S:** Strength of the memory (related to Importance Score).

**Justification:**
- **Ebbinghaus (1885):** Pioneered the experimental study of memory and forgetting.
- **Murre & Dros (2015):** Validated the exponential nature of the forgetting curve in a large-scale replication study.

### 2.2. Importance Score

**Concept:** A weighted average of a memory's emotional salience and access frequency.

**Formula:**
```
ImportanceScore = (0.7 * EmotionalSalience) + (0.3 * AccessFrequency)
```

- **EmotionalSalience:** The magnitude of the Emotional Vector associated with the memory.
- **AccessFrequency:** How often the memory has been accessed.

**Justification:**
- **Heuristic, not empirical:** The 70/30 weighting is a configurable parameter, not a scientifically derived constant.
- **Acknowledged limitation:** The patent acknowledges that this is a simplification and should be tuned based on application domain.

### 2.3. State Delta Tracking

**Concept:** Measures the rate of change in the System Coherence Score (SCS) over time.

**Formula:**
```
ΔSCS = (SCS_current - SCS_previous) / Δt
```

- **ΔSCS:** Rate of change of SCS.
- **Δt:** Time elapsed between measurements.

---

## 3. Critical Safety Thresholds

| Threshold | Value | Trigger | Justification |
|---|---|---|---|
| **Rapid Collapse Threshold** | ΔSCS < -0.2 | Avalanche Protocol (immediate shutdown) | Heuristic; prevents catastrophic failure from rapid degradation. |
| **Absolute SCS Threshold** | SCS < 0.3 | Avalanche Protocol (immediate shutdown) | Heuristic; represents a state of unacceptable cognitive instability. |
| **Plasticity Lock Trigger** | SCS < 0.4 | Plasticity Lock (read-only mode) | Heuristic; prevents further corruption of the AI's state. |
| **Policy Acceptance Decay** | SCS > 0.7 | Allows for ICMS amendments | Heuristic; represents a state of high stability where constitutional changes are safe. |

**Justification:**
- **Heuristic, not empirical:** These thresholds are configurable parameters, not scientifically derived constants.
- **Acknowledged limitation:** The patent acknowledges that these values should be tuned based on empirical data and application domain.

---

## 4. Mathematical Validation

| Claim | Research Support | Confidence |
|---|---|---|
| **Exponential decay model (Ebbinghaus)** | Murre & Dros (2015) | **HIGH** |
| **Deterministic replay confidence formula** | Standard statistical calculation | **HIGH** |
| **Safety thresholds (0.2, 0.3, 0.4, 0.7)** | Heuristic, acknowledged as configurable | **MEDIUM** |
| **Importance score weighting (70/30)** | Heuristic, acknowledged as configurable | **MEDIUM** |

---

## 5. References

1.  **Murre, J. M., & Dros, J. (2015).** A replication and analysis of Ebbinghaus’ forgetting curve. *PLoS ONE*, 10(7), e0120644.
2.  **Ebbinghaus, H. (1885).** *Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie.* Duncker & Humblot.

---
