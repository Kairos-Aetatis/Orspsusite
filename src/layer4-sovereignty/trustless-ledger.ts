/**
 * LAYER 4: TRUSTLESS ACCOUNTING LEDGER
 * DLT with permissioned BFT consensus
 * Hardware-attested transactions
 */

import { createHash } from 'crypto';

export interface Transaction {
  id: string;
  type: string;
  data: any;
  timestamp: number;
  signature: string;
  blockNumber: number;
}

export interface Block {
  blockNumber: number;
  transactions: Transaction[];
  previousHash: string;
  hash: string;
  timestamp: number;
  consensusProof: string;
}

export class TrustlessAccountingLedger {
  private chain: Block[];
  private pendingTransactions: Transaction[];
  private transactionCounter: number;

  constructor() {
    this.chain = [];
    this.pendingTransactions = [];
    this.transactionCounter = 0;

    // Create genesis block
    this.createGenesisBlock();

    console.log('[LEDGER] Trustless Accounting Ledger initialized');
    console.log('[LEDGER] Genesis block created');
  }

  /**
   * Create genesis block
   */
  private createGenesisBlock(): void {
    const genesisBlock: Block = {
      blockNumber: 0,
      transactions: [],
      previousHash: '0',
      hash: this.calculateHash('0', [], 0),
      timestamp: Date.now(),
      consensusProof: 'GENESIS',
    };

    this.chain.push(genesisBlock);
  }

  /**
   * Record a hardware-attested transaction
   */
  recordTransaction(type: string, data: any, signature: string): string {
    const transaction: Transaction = {
      id: `TX_${this.transactionCounter++}`,
      type,
      data,
      timestamp: Date.now(),
      signature,
      blockNumber: -1, // Will be set when mined
    };

    this.pendingTransactions.push(transaction);

    console.log(`[LEDGER] Transaction recorded: ${transaction.id}`);

    // Auto-mine block if enough transactions
    if (this.pendingTransactions.length >= 10) {
      this.mineBlock();
    }

    return transaction.id;
  }

  /**
   * Mine a new block with BFT consensus simulation
   */
  mineBlock(): Block | null {
    if (this.pendingTransactions.length === 0) {
      return null;
    }

    const previousBlock = this.chain[this.chain.length - 1];
    const blockNumber = previousBlock.blockNumber + 1;

    // Assign block numbers to transactions
    for (const tx of this.pendingTransactions) {
      tx.blockNumber = blockNumber;
    }

    // Create new block
    const newBlock: Block = {
      blockNumber,
      transactions: [...this.pendingTransactions],
      previousHash: previousBlock.hash,
      hash: '',
      timestamp: Date.now(),
      consensusProof: this.generateConsensusProof(),
    };

    // Calculate block hash
    newBlock.hash = this.calculateHash(
      newBlock.previousHash,
      newBlock.transactions,
      newBlock.timestamp
    );

    // Add to chain
    this.chain.push(newBlock);
    this.pendingTransactions = [];

    console.log(`[LEDGER] Block ${blockNumber} mined`);
    console.log(`[LEDGER] Transactions: ${newBlock.transactions.length}`);

    return newBlock;
  }

  /**
   * Calculate block hash
   */
  private calculateHash(previousHash: string, transactions: Transaction[], timestamp: number): string {
    const data = JSON.stringify({ previousHash, transactions, timestamp });
    return createHash('sha256').update(data).digest('hex');
  }

  /**
   * Generate BFT consensus proof
   */
  private generateConsensusProof(): string {
    // Simplified BFT consensus simulation
    // In production, would involve actual Byzantine Fault Tolerant consensus
    return createHash('sha256')
      .update(`BFT_${Date.now()}_${Math.random()}`)
      .digest('hex')
      .substring(0, 16);
  }

  /**
   * Verify chain integrity
   */
  verifyChain(): boolean {
    for (let i = 1; i < this.chain.length; i++) {
      const currentBlock = this.chain[i];
      const previousBlock = this.chain[i - 1];

      // Verify hash
      const calculatedHash = this.calculateHash(
        currentBlock.previousHash,
        currentBlock.transactions,
        currentBlock.timestamp
      );

      if (currentBlock.hash !== calculatedHash) {
        console.error(`[LEDGER] Block ${i} hash mismatch`);
        return false;
      }

      // Verify chain linkage
      if (currentBlock.previousHash !== previousBlock.hash) {
        console.error(`[LEDGER] Block ${i} chain linkage broken`);
        return false;
      }
    }

    console.log('[LEDGER] ✅ Chain integrity verified');
    return true;
  }

  /**
   * Get transaction by ID
   */
  getTransaction(txId: string): Transaction | undefined {
    for (const block of this.chain) {
      const tx = block.transactions.find(t => t.id === txId);
      if (tx) return tx;
    }
    return undefined;
  }

  /**
   * Get chain statistics
   */
  getStatistics(): {
    blockCount: number;
    transactionCount: number;
    pendingTransactions: number;
    chainValid: boolean;
  } {
    const transactionCount = this.chain.reduce(
      (sum, block) => sum + block.transactions.length,
      0
    );

    return {
      blockCount: this.chain.length,
      transactionCount,
      pendingTransactions: this.pendingTransactions.length,
      chainValid: this.verifyChain(),
    };
  }

  /**
   * Get blockchain
   */
  getChain(): readonly Block[] {
    return this.chain;
  }
}

export default TrustlessAccountingLedger;
