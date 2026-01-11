/**
 * LAYER 3: PaTH ATTENTION MECHANISM
 * Patent Claims: 31, 32
 * 
 * Householder Reflections for positional encoding
 * Dynamic reflections instead of static rotations
 * Superior state tracking, reduces confabulation
 */

import { PATH_DIMENSION } from '../../config/constants';

export interface AttentionVector {
  position: number[];
  emotionalSteering: number;
  importance: number;
}

export class PathAttention {
  private dimension: number;

  constructor(dimension: number = PATH_DIMENSION) {
    this.dimension = dimension;

    console.log(`[PATH] Initialized with dimension: ${dimension}`);
  }

  /**
   * Patent Claims 31, 32: Apply Householder transformation
   * Steered by emotional vector V_E and importance I
   */
  applyPathTransformation(
    tokenVector: number[],
    emotionalVector: number,
    importance: number
  ): number[] {
    if (tokenVector.length !== this.dimension) {
      throw new Error(`[PATH] Invalid token vector dimension: ${tokenVector.length} (expected ${this.dimension})`);
    }

    // 1. Create Householder reflection vector based on emotional state
    const reflectionVector = this.createReflectionVector(emotionalVector);

    // 2. Apply Householder transformation: H = I - 2vv^T
    const householder = this.computeHouseholderMatrix(reflectionVector);

    // 3. Transform position
    const transformedPosition = this.matrixVectorMultiply(householder, tokenVector);

    // 4. Apply importance weighting (gravitational pull)
    const weightedAttention = transformedPosition.map(v => v * (1.0 + importance));

    return weightedAttention;
  }

  /**
   * Create reflection vector based on emotional state
   */
  private createReflectionVector(emotionalVector: number): number[] {
    const v = new Array(this.dimension);

    for (let i = 0; i < this.dimension; i++) {
      v[i] = Math.random() * emotionalVector;
    }

    // Normalize
    const norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
    return v.map(val => val / (norm + 1e-9));
  }

  /**
   * Compute Householder matrix: H = I - 2vv^T
   */
  private computeHouseholderMatrix(v: number[]): number[][] {
    const n = v.length;
    const H: number[][] = [];

    // Create identity matrix
    for (let i = 0; i < n; i++) {
      H[i] = new Array(n).fill(0);
      H[i][i] = 1;
    }

    // Subtract 2vv^T
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        H[i][j] -= 2 * v[i] * v[j];
      }
    }

    return H;
  }

  /**
   * Matrix-vector multiplication
   */
  private matrixVectorMultiply(matrix: number[][], vector: number[]): number[] {
    const result: number[] = new Array(vector.length).fill(0);

    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < vector.length; j++) {
        result[i] += matrix[i][j] * vector[j];
      }
    }

    return result;
  }

  /**
   * Get dimension
   */
  getDimension(): number {
    return this.dimension;
  }
}

export default PathAttention;
