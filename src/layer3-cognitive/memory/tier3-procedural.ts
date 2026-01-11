/**
 * LAYER 3: MEMORY TIER 3 - PROCEDURAL CODEX (Rust Mechanic)
 * Patent Claim: Procedural memory with skill decay
 * 
 * Separates immutable `action_script` from mutable `proficiency_score`
 * Skills can "rust" without losing logical integrity
 */

export interface ProceduralSkill {
  id: string;
  name: string;
  actionScript: string;        // Immutable: How to perform the skill
  proficiencyScore: number;    // Mutable: How well we can perform it
  lastPracticed: number;
  practiceCount: number;
  rustRate: number;            // How quickly skill decays
  timestamp: number;
}

export class ProceduralCodex {
  private skills: Map<string, ProceduralSkill>;
  private currentTime: number;

  constructor() {
    this.skills = new Map();
    this.currentTime = 0;

    console.log('[PROCEDURAL] Initialized with Rust Mechanic');
  }

  /**
   * Learn a new procedural skill
   * Action script is immutable, proficiency starts low
   */
  learnSkill(name: string, actionScript: string, initialProficiency: number = 0.5): string {
    const id = this.generateSkillId(name);

    const skill: ProceduralSkill = {
      id,
      name,
      actionScript,                      // Immutable
      proficiencyScore: initialProficiency, // Mutable
      lastPracticed: this.currentTime,
      practiceCount: 1,
      rustRate: 0.01,                     // Default rust rate
      timestamp: Date.now(),
    };

    this.skills.set(id, skill);

    console.log(`[PROCEDURAL] Learned skill: ${name}`);
    console.log(`  Initial proficiency: ${initialProficiency.toFixed(2)}`);

    return id;
  }

  /**
   * Practice a skill (increases proficiency)
   */
  practiceSkill(skillId: string): boolean {
    const skill = this.skills.get(skillId);

    if (!skill) {
      console.error(`[PROCEDURAL] Unknown skill: ${skillId}`);
      return false;
    }

    // Increase proficiency (with diminishing returns)
    const improvement = (1 - skill.proficiencyScore) * 0.1;
    skill.proficiencyScore = Math.min(1.0, skill.proficiencyScore + improvement);
    skill.lastPracticed = this.currentTime;
    skill.practiceCount++;

    console.log(`[PROCEDURAL] Practiced ${skill.name}`);
    console.log(`  Proficiency: ${skill.proficiencyScore.toFixed(3)} (+${improvement.toFixed(3)})`);

    return true;
  }

  /**
   * Apply rust decay to all skills over time
   * Skills decay if not practiced
   */
  tick(timeSteps: number = 1): void {
    this.currentTime += timeSteps;

    for (const [id, skill] of this.skills) {
      const timeSinceP ractice = this.currentTime - skill.lastPracticed;

      if (timeSinceP ractice > 0) {
        // Apply exponential decay
        const decay = skill.rustRate * timeSinceP ractice;
        const newProficiency = Math.max(0.1, skill.proficiencyScore - decay);

        if (newProficiency < skill.proficiencyScore) {
          const rustAmount = skill.proficiencyScore - newProficiency;
          skill.proficiencyScore = newProficiency;

          if (this.currentTime % 100 === 0 && rustAmount > 0.001) {
            console.log(`[PROCEDURAL] ${skill.name} rusting: ${skill.proficiencyScore.toFixed(3)}`);
          }
        }
      }
    }
  }

  /**
   * Execute a skill
   * Returns success probability based on proficiency
   */
  executeSkill(skillId: string): { success: boolean; proficiency: number; script: string } {
    const skill = this.skills.get(skillId);

    if (!skill) {
      throw new Error(`[PROCEDURAL] Cannot execute unknown skill: ${skillId}`);
    }

    // Success probability based on proficiency
    const success = Math.random() < skill.proficiencyScore;

    // Executing counts as practice
    this.practiceSkill(skillId);

    console.log(`[PROCEDURAL] Executed ${skill.name}: ${success ? 'SUCCESS' : 'FAILURE'}`);

    return {
      success,
      proficiency: skill.proficiencyScore,
      script: skill.actionScript,
    };
  }

  /**
   * Get skill by ID
   */
  getSkill(skillId: string): ProceduralSkill | undefined {
    return this.skills.get(skillId);
  }

  /**
   * Get all skills
   */
  getAllSkills(): ProceduralSkill[] {
    return Array.from(this.skills.values());
  }

  /**
   * Get skills sorted by proficiency
   */
  getSkillsByProficiency(): ProceduralSkill[] {
    return this.getAllSkills().sort((a, b) => b.proficiencyScore - a.proficiencyScore);
  }

  /**
   * Get rusty skills (proficiency below threshold)
   */
  getRustySkills(threshold: number = 0.5): ProceduralSkill[] {
    return this.getAllSkills().filter(s => s.proficiencyScore < threshold);
  }

  /**
   * Update rust rate for a skill
   */
  setRustRate(skillId: string, rustRate: number): void {
    const skill = this.skills.get(skillId);

    if (!skill) {
      throw new Error(`[PROCEDURAL] Unknown skill: ${skillId}`);
    }

    skill.rustRate = rustRate;
    console.log(`[PROCEDURAL] Updated rust rate for ${skill.name}: ${rustRate}`);
  }

  /**
   * Generate skill ID
   */
  private generateSkillId(name: string): string {
    return `SKILL_${name.replace(/\s+/g, '_')}_${Date.now()}`;
  }

  /**
   * Get statistics
   */
  getStatistics(): {
    totalSkills: number;
    averageProficiency: number;
    rustySkills: number;
    masteredSkills: number;
    currentTime: number;
  } {
    const skills = this.getAllSkills();

    const averageProficiency = skills.length > 0
      ? skills.reduce((sum, s) => sum + s.proficiencyScore, 0) / skills.length
      : 0;

    const rustySkills = skills.filter(s => s.proficiencyScore < 0.5).length;
    const masteredSkills = skills.filter(s => s.proficiencyScore > 0.9).length;

    return {
      totalSkills: skills.length,
      averageProficiency,
      rustySkills,
      masteredSkills,
      currentTime: this.currentTime,
    };
  }

  /**
   * Get current time
   */
  getCurrentTime(): number {
    return this.currentTime;
  }
}

export default ProceduralCodex;
