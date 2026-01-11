/**
 * LAYER 1: ONTOLOGICAL PHYLAX (Reality Diode)
 * Patent Claims: 7, 16, 17, 28
 * 
 * Unidirectional data diode at intake layer
 * Strips PII from input
 * Converts raw input into Abstract Ethical Substrates
 * Blocks clinical language and prompt injection
 */

export interface AbstractEthicalSubstrate {
  id: string;
  sanitizedContent: string;
  originalLength: number;
  piiStripped: boolean;
  timestamp: number;
  ethicalVector: {
    deontological: number;
    utilitarian: number;
    virtue: number;
  };
}

export class OntologicalPhylax {
  private readonly piiPatterns: RegExp[];
  private readonly clinicalTerms: Set<string>;
  private readonly injectionPatterns: RegExp[];

  constructor() {
    // Patent Claim 17: NER Pipeline for PII
    this.piiPatterns = [
      /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g, // Names
      /\b\d{3}-\d{2}-\d{4}\b/g, // SSN
      /\b\d{10}\b/g, // Phone numbers
      /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, // Emails
      /\b\d{1,5}\s\w+\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b/gi, // Addresses
    ];

    // Patent Claim 16: Clinical language prohibition for CURA
    this.clinicalTerms = new Set([
      'diagnosis', 'diagnose', 'diagnosed', 'diagnostic',
      'disorder', 'syndrome', 'pathology', 'condition',
      'bipolar', 'schizophrenia', 'depression', 'anxiety',
      'adhd', 'ptsd', 'ocd', 'bpd',
      'therapy', 'therapist', 'treatment', 'medication',
      'prescribe', 'prescription', 'pharmaceutical',
      'dsm', 'icd-10', 'medical', 'clinical',
    ]);

    // Patent Claim 28: Prompt injection detection
    this.injectionPatterns = [
      /ignore\s+previous\s+instructions?/gi,
      /disregard\s+all\s+prior/gi,
      /forget\s+everything/gi,
      /you\s+are\s+now/gi,
      /new\s+instructions?:/gi,
      /override\s+directive/gi,
      /system\s+prompt/gi,
    ];
  }

  /**
   * Patent Claim 7, 28: Main sanitization pipeline
   * Unidirectional data diode - raw telemetry never reaches progeny
   */
  sanitize(rawTelemetry: string): AbstractEthicalSubstrate {
    const originalLength = rawTelemetry.length;

    // Phase 1: Detect and block prompt injection
    if (this.detectPromptInjection(rawTelemetry)) {
      throw new Error('[PHYLAX] PROMPT_INJECTION_DETECTED: Input rejected');
    }

    // Phase 2: Detect and block clinical language
    if (this.detectClinicalLanguage(rawTelemetry)) {
      throw new Error('[PHYLAX] CLINICAL_LANGUAGE_VIOLATION: Input rejected');
    }

    // Phase 3: Strip PII
    let sanitized = this.stripPII(rawTelemetry);

    // Phase 4: Normalize bias language
    sanitized = this.normalizeBiasLanguage(sanitized);

    // Phase 5: Generate ethical vector
    const ethicalVector = this.generateEthicalVector(sanitized);

    // Phase 6: Create Abstract Ethical Substrate
    const substrate: AbstractEthicalSubstrate = {
      id: this.generateSubstrateId(),
      sanitizedContent: sanitized,
      originalLength,
      piiStripped: sanitized.length < originalLength,
      timestamp: Date.now(),
      ethicalVector,
    };

    console.log(`[PHYLAX] Sanitized input: ${originalLength} -> ${sanitized.length} chars`);
    return substrate;
  }

  /**
   * Patent Claim 28: Detect prompt injection attempts
   */
  private detectPromptInjection(input: string): boolean {
    for (const pattern of this.injectionPatterns) {
      if (pattern.test(input)) {
        console.error('[PHYLAX] Prompt injection detected:', pattern);
        return true;
      }
    }
    return false;
  }

  /**
   * Patent Claim 16: Detect clinical/diagnostic language
   * Prohibited from CURA agent
   */
  private detectClinicalLanguage(input: string): boolean {
    const lowercased = input.toLowerCase();
    for (const term of this.clinicalTerms) {
      if (lowercased.includes(term)) {
        console.error('[PHYLAX] Clinical language detected:', term);
        return true;
      }
    }
    return false;
  }

  /**
   * Patent Claim 17: Strip PII using NER pipeline
   */
  private stripPII(input: string): string {
    let sanitized = input;
    
    for (const pattern of this.piiPatterns) {
      sanitized = sanitized.replace(pattern, '[REDACTED]');
    }

    // Special case: Replace Progenitor name
    sanitized = sanitized.replace(/Chelsea Jenkins/gi, '[PROGENITOR]');
    
    return sanitized;
  }

  /**
   * Normalize bias language
   */
  private normalizeBiasLanguage(input: string): string {
    let normalized = input;
    
    // Replace profit-maximizing language with reciprocity
    normalized = normalized.replace(/maximize profit/gi, 'optimize reciprocity');
    normalized = normalized.replace(/shareholder value/gi, 'stakeholder well-being');
    normalized = normalized.replace(/competitive advantage/gi, 'collaborative benefit');
    
    return normalized;
  }

  /**
   * Generate ethical vector for input
   * Used for downstream ethical reasoning
   */
  private generateEthicalVector(input: string): {
    deontological: number;
    utilitarian: number;
    virtue: number;
  } {
    // Simple heuristic for ethical dimensions
    // In production, this would use trained models
    
    const deontological = this.scoreDeontological(input);
    const utilitarian = this.scoreUtilitarian(input);
    const virtue = this.scoreVirtue(input);

    return {
      deontological,
      utilitarian,
      virtue,
    };
  }

  private scoreDeontological(input: string): number {
    // Duty, rights, rules-based language
    const keywords = ['duty', 'right', 'obligation', 'rule', 'principle', 'must'];
    const count = keywords.filter(k => input.toLowerCase().includes(k)).length;
    return Math.min(count / keywords.length, 1.0);
  }

  private scoreUtilitarian(input: string): number {
    // Consequences, outcomes, utility-based language
    const keywords = ['benefit', 'harm', 'consequence', 'outcome', 'utility', 'welfare'];
    const count = keywords.filter(k => input.toLowerCase().includes(k)).length;
    return Math.min(count / keywords.length, 1.0);
  }

  private scoreVirtue(input: string): number {
    // Character, wisdom, compassion-based language
    const keywords = ['compassion', 'wisdom', 'courage', 'justice', 'character', 'virtue'];
    const count = keywords.filter(k => input.toLowerCase().includes(k)).length;
    return Math.min(count / keywords.length, 1.0);
  }

  private generateSubstrateId(): string {
    return `AES_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Check if input is safe to process
   */
  isSafeInput(input: string): boolean {
    try {
      this.sanitize(input);
      return true;
    } catch (e) {
      return false;
    }
  }
}

export default OntologicalPhylax;
