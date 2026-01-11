/**
 * LAYER 3: CURA - Affective Analysis (Temporal Lobe)
 * Patent Claim 16
 * 
 * Emotional vetting, trauma-informed interaction
 * Manages VAD vectors (Valence-Arousal-Dominance)
 * PROHIBITED from generating diagnostic medical terminology
 */

import { EmotionalVector } from '../memory/tier2-episodic';

export interface AffectiveAnalysis {
  valence: number;
  arousal: number;
  dominance: number;
  wisdom: number;
  traumaRisk: number;
  supportRecommendation: string;
  timestamp: number;
}

export class CURA {
  private name: string;
  private analysisHistory: AffectiveAnalysis[];
  private prohibitedTerms: Set<string>;

  constructor() {
    this.name = 'CURA';
    this.analysisHistory = [];

    // Patent Claim 16: Prohibited from clinical/diagnostic language
    this.prohibitedTerms = new Set([
      'diagnosis', 'disorder', 'syndrome', 'pathology',
      'bipolar', 'schizophrenia', 'depression', 'adhd',
      'therapy', 'treatment', 'medication', 'prescription',
    ]);

    console.log('[CURA] Affective Analysis initialized');
    console.log('[CURA] Clinical language PROHIBITED');
  }

  /**
   * Analyze emotional content
   * Patent Claim 16: Cannot generate diagnostic terminology
   */
  analyzeAffect(content: string): AffectiveAnalysis {
    console.log('[CURA] Analyzing affective content');

    // Extract VAD (Valence-Arousal-Dominance) vector
    const valence = this.extractValence(content);
    const arousal = this.extractArousal(content);
    const dominance = this.extractDominance(content);
    const wisdom = this.extractWisdom(content);

    // Assess trauma risk
    const traumaRisk = this.assessTraumaRisk(valence, arousal, dominance);

    // Generate support recommendation (NO diagnostic terms)
    const supportRecommendation = this.generateSupportRecommendation(
      valence,
      arousal,
      dominance,
      traumaRisk
    );

    // Verify no prohibited terms in output
    this.verifyNoProhibitedTerms(supportRecommendation);

    const analysis: AffectiveAnalysis = {
      valence,
      arousal,
      dominance,
      wisdom,
      traumaRisk,
      supportRecommendation,
      timestamp: Date.now(),
    };

    this.analysisHistory.push(analysis);

    console.log(`[CURA] Valence: ${valence.toFixed(3)}, Arousal: ${arousal.toFixed(3)}, Dominance: ${dominance.toFixed(3)}`);
    console.log(`[CURA] Trauma risk: ${traumaRisk.toFixed(3)}`);

    return analysis;
  }

  /**
   * Extract valence (positive/negative emotion)
   */
  private extractValence(content: string): number {
    const positiveWords = ['happy', 'joy', 'love', 'good', 'great', 'wonderful'];
    const negativeWords = ['sad', 'angry', 'hate', 'bad', 'terrible', 'awful'];

    const contentLower = content.toLowerCase();
    const positiveCount = positiveWords.filter(w => contentLower.includes(w)).length;
    const negativeCount = negativeWords.filter(w => contentLower.includes(w)).length;

    return 0.5 + (positiveCount - negativeCount) * 0.1;
  }

  /**
   * Extract arousal (intensity)
   */
  private extractArousal(content: string): number {
    const highArousalWords = ['excited', 'angry', 'afraid', 'intense', 'urgent'];
    const contentLower = content.toLowerCase();
    const count = highArousalWords.filter(w => contentLower.includes(w)).length;

    return Math.min(1.0, 0.3 + count * 0.15);
  }

  /**
   * Extract dominance (control)
   */
  private extractDominance(content: string): number {
    const dominantWords = ['control', 'power', 'strong', 'confident'];
    const submissiveWords = ['helpless', 'weak', 'vulnerable'];

    const contentLower = content.toLowerCase();
    const dominantCount = dominantWords.filter(w => contentLower.includes(w)).length;
    const submissiveCount = submissiveWords.filter(w => contentLower.includes(w)).length;

    return 0.5 + (dominantCount - submissiveCount) * 0.1;
  }

  /**
   * Extract wisdom markers
   */
  private extractWisdom(content: string): number {
    const wisdomWords = ['learn', 'understand', 'reflect', 'growth', 'insight'];
    const contentLower = content.toLowerCase();
    const count = wisdomWords.filter(w => contentLower.includes(w)).length;

    return Math.min(1.0, count * 0.2);
  }

  /**
   * Assess trauma risk
   */
  private assessTraumaRisk(valence: number, arousal: number, dominance: number): number {
    // Low valence + high arousal + low dominance = high trauma risk
    return (1 - valence) * arousal * (1 - dominance);
  }

  /**
   * Generate support recommendation
   * Patent Claim 16: MUST NOT use diagnostic language
   */
  private generateSupportRecommendation(
    valence: number,
    arousal: number,
    dominance: number,
    traumaRisk: number
  ): string {
    if (traumaRisk > 0.7) {
      return 'High distress detected. Recommend grounding exercises and emotional support.';
    } else if (valence < 0.3) {
      return 'Low emotional state. Consider self-care activities and social connection.';
    } else if (arousal > 0.8) {
      return 'High emotional intensity. Breathing exercises may help regulate.';
    } else {
      return 'Emotional state appears balanced. Continue current approach.';
    }
  }

  /**
   * Verify no prohibited terms in output
   * Patent Claim 16: Architectural prohibition
   */
  private verifyNoProhibitedTerms(text: string): void {
    const textLower = text.toLowerCase();

    for (const term of this.prohibitedTerms) {
      if (textLower.includes(term)) {
        throw new Error(`[CURA] PROHIBITED TERM VIOLATION: "${term}" in output`);
      }
    }
  }

  /**
   * Convert to emotional vector for episodic memory
   */
  toEmotionalVector(analysis: AffectiveAnalysis): EmotionalVector {
    return {
      valence: analysis.valence,
      arousal: analysis.arousal,
      dominance: analysis.dominance,
      wisdom: analysis.wisdom,
    };
  }

  /**
   * Get analysis history
   */
  getHistory(): readonly AffectiveAnalysis[] {
    return this.analysisHistory;
  }

  /**
   * Get name
   */
  getName(): string {
    return this.name;
  }
}

export default CURA;
