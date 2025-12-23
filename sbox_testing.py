"""
S-box Cryptographic Strength Testing Module (FIXED VERSION)
Implements all cryptographic tests: NL, SAC, BIC-NL, BIC-SAC, LAP, DAP

MAJOR FIX: BIC-SAC calculation now correctly tests bit independence
- Old version: counted when BOTH bits change (wrong)
- New version: counts when bits change INDEPENDENTLY (XOR logic - correct)
"""

import numpy as np
from typing import Dict, Tuple
from itertools import combinations

class SBoxCryptoTest:
    """Cryptographic strength testing for S-boxes"""
    
    def __init__(self, sbox: np.ndarray):
        """
        Initialize with S-box
        
        Args:
            sbox: 16x16 S-box matrix
        """
        self.sbox = sbox.flatten()
        self.n = 8  # bit length
        self.N = 256  # 2^n
    
    def walsh_hadamard_transform(self, f: np.ndarray) -> np.ndarray:
        """Compute Walsh-Hadamard Transform"""
        n = len(f)
        result = np.zeros(n, dtype=np.int32)
        
        for w in range(n):
            s = 0
            for x in range(n):
                s += (-1) ** (self._dot_product(x, w) ^ int(f[x]))
            result[w] = s
        
        return result
    
    def _dot_product(self, a: int, b: int) -> int:
        """Binary dot product"""
        result = 0
        while a and b:
            if a & 1 and b & 1:
                result ^= 1
            a >>= 1
            b >>= 1
        return result
    
    def test_nonlinearity(self) -> Tuple[float, Dict]:
        """
        Test 1: Nonlinearity (NL)
        
        NL(f) = min d(f(x), g(x)) where g(x) is affine
        Higher is better, max = 2^(n-1) - 2^(n/2-1) = 112 for n=8
        
        Returns:
            (NL_value, details)
        """
        nl_values = []
        
        for bit in range(8):
            # Extract bit function
            f = np.array([(self.sbox[x] >> bit) & 1 for x in range(self.N)])
            
            # Walsh-Hadamard transform
            W = self.walsh_hadamard_transform(f)
            
            # NL for this bit
            max_walsh = np.max(np.abs(W))
            nl = (self.N - max_walsh) // 2
            nl_values.append(nl)
        
        avg_nl = np.mean(nl_values)
        min_nl = np.min(nl_values)
        
        details = {
            'average': float(avg_nl),
            'minimum': int(min_nl),
            'per_bit': [int(x) for x in nl_values],
            'ideal': 112,
            'score': (min_nl / 112) * 100
        }
        
        return min_nl, details
    
    def test_sac(self) -> Tuple[float, Dict]:
        """
        Test 2: Strict Avalanche Criterion (SAC)
        
        SAC measures output bit changes when flipping one input bit
        Ideal value = 0.5
        
        Formula: For each input bit i and output bit j:
        SAC(i,j) = (1/256) * Σ[f_j(x) ⊕ f_j(x ⊕ e_i)]
        where e_i is the unit vector with bit i set
        
        Returns:
            (SAC_value, details)
        """
        sac_matrix = np.zeros((8, 8))
        
        for i in range(8):  # Input bit position
            for j in range(8):  # Output bit position
                changes = 0
                for x in range(self.N):
                    # Flip bit i
                    x_flipped = x ^ (1 << i)
                    
                    # Check if output bit j changes
                    out1 = (self.sbox[x] >> j) & 1
                    out2 = (self.sbox[x_flipped] >> j) & 1
                    
                    if out1 != out2:
                        changes += 1
                
                sac_matrix[i, j] = changes / self.N
        
        avg_sac = np.mean(sac_matrix)
        deviation = abs(avg_sac - 0.5)
        
        details = {
            'average': float(avg_sac),
            'deviation_from_ideal': float(deviation),
            'matrix': sac_matrix.tolist(),
            'ideal': 0.5,
            'score': (1 - deviation * 2) * 100
        }
        
        return avg_sac, details
    
    def test_bic_nl(self) -> Tuple[float, Dict]:
        """
        Test 3: Bit Independence Criterion - Nonlinearity (BIC-NL)
        
        Tests nonlinearity between output bit pairs
        
        Returns:
            (BIC_NL_value, details)
        """
        nl_values = []
        
        for i, j in combinations(range(8), 2):
            # XOR of two output bits
            f = np.array([((self.sbox[x] >> i) & 1) ^ ((self.sbox[x] >> j) & 1) 
                         for x in range(self.N)])
            
            # Walsh-Hadamard transform
            W = self.walsh_hadamard_transform(f)
            
            # NL for this pair
            max_walsh = np.max(np.abs(W))
            nl = (self.N - max_walsh) // 2
            nl_values.append(nl)
        
        avg_nl = np.mean(nl_values)
        min_nl = np.min(nl_values)
        
        details = {
            'average': float(avg_nl),
            'minimum': int(min_nl),
            'pairs_tested': len(nl_values),
            'ideal': 112,
            'score': (min_nl / 112) * 100
        }
        
        return min_nl, details
    
    def test_bic_sac(self) -> Tuple[float, Dict]:
        """
        Test 4: Bit Independence Criterion - SAC (BIC-SAC) - FIXED VERSION
        
        Tests independence between output bit changes when input bits are flipped.
        
        CORRECTED FORMULA:
        For each pair of output bits (i, j), measure:
        P(bit_i changes XOR bit_j changes) = should be close to 0.5
        
        This tests if the two output bits change independently when input bits flip.
        
        Formula from standard cryptographic literature:
        BIC-SAC(i,j) = (1/(8*256)) * Σ_k Σ_x [(f_i(x) ⊕ f_i(x⊕e_k)) ⊕ (f_j(x) ⊕ f_j(x⊕e_k))]
        
        The XOR of the two changes should be 1 with probability 0.5 if bits are independent.
        
        Returns:
            (BIC_SAC_value, details)
        """
        # Create full 8x8 matrix (diagonal will be set to -1 for display)
        bic_sac_matrix = np.zeros((8, 8))
        
        # For each pair of output bits (including i=j for completeness)
        for i in range(8):
            for j in range(8):
                if i == j:
                    # Diagonal elements are not calculated (same bit can't be independent of itself)
                    bic_sac_matrix[i, j] = -1  # Mark as invalid
                    continue
                
                independence_count = 0
                total_count = 0
                
                # For each input bit flip
                for k in range(8):
                    for x in range(self.N):
                        x_flipped = x ^ (1 << k)
                        
                        # Check if bit i changes
                        bit_i_original = (self.sbox[x] >> i) & 1
                        bit_i_flipped = (self.sbox[x_flipped] >> i) & 1
                        bit_i_changed = (bit_i_original != bit_i_flipped)
                        
                        # Check if bit j changes
                        bit_j_original = (self.sbox[x] >> j) & 1
                        bit_j_flipped = (self.sbox[x_flipped] >> j) & 1
                        bit_j_changed = (bit_j_original != bit_j_flipped)
                        
                        # Count when exactly ONE of the two bits changes (XOR = 1)
                        # This tests independence: if independent, P(XOR=1) ≈ 0.5
                        if bit_i_changed != bit_j_changed:  # XOR operation
                            independence_count += 1
                        
                        total_count += 1
                
                # Calculate probability
                bic_sac_matrix[i, j] = independence_count / total_count
        
        # Get values excluding diagonal for statistics
        values = []
        for i in range(8):
            for j in range(8):
                if i != j:
                    values.append(bic_sac_matrix[i, j])
        
        avg_bic_sac = np.mean(values)
        deviation = abs(avg_bic_sac - 0.5)
        
        # For display, replace -1 with None in the matrix copy
        display_matrix = []
        for i in range(8):
            row = []
            for j in range(8):
                if i == j:
                    row.append(None)  # Will display as "-" or empty
                else:
                    row.append(float(bic_sac_matrix[i, j]))
            display_matrix.append(row)
        
        details = {
            'average': float(avg_bic_sac),
            'deviation_from_ideal': float(deviation),
            'matrix': display_matrix,
            'ideal': 0.5,
            'score': (1 - deviation * 2) * 100,
            'note': 'Fixed: Now correctly measures bit independence using XOR logic'
        }
        
        return avg_bic_sac, details
    
    def test_lap(self) -> Tuple[float, Dict]:
        """
        Test 5: Linear Approximation Probability (LAP)
        
        LAP = max |#{x: x·Cx = S(x)·Cy} - 2^(n-1)| / 2^n
        Lower is better, ideal = 0.0625 for 8-bit S-boxes
        
        Returns:
            (LAP_value, details)
        """
        max_bias = 0
        best_masks = (0, 0)
        
        for input_mask in range(1, self.N):
            for output_mask in range(1, self.N):
                count = 0
                
                for x in range(self.N):
                    input_parity = self._dot_product(x, input_mask)
                    output_parity = self._dot_product(self.sbox[x], output_mask)
                    
                    if input_parity == output_parity:
                        count += 1
                
                bias = abs(count - self.N // 2)
                if bias > max_bias:
                    max_bias = bias
                    best_masks = (input_mask, output_mask)
        
        lap = max_bias / self.N
        
        details = {
            'value': float(lap),
            'max_bias': int(max_bias),
            'best_input_mask': hex(best_masks[0]),
            'best_output_mask': hex(best_masks[1]),
            'ideal': 0.0625,
            'score': (0.0625 / lap) * 100 if lap > 0 else 100
        }
        
        return lap, details
    
    def test_dap(self) -> Tuple[float, Dict]:
        """
        Test 6: Differential Approximation Probability (DAP)
        
        DAP = max #{x: S(x) ⊕ S(x⊕Δx) = Δy} / 2^n
        Lower is better, ideal = 0.015625 for 8-bit S-boxes
        
        Returns:
            (DAP_value, details)
        """
        max_count = 0
        best_diffs = (0, 0)
        
        for dx in range(1, self.N):
            diff_table = {}
            
            for x in range(self.N):
                dy = self.sbox[x] ^ self.sbox[x ^ dx]
                diff_table[dy] = diff_table.get(dy, 0) + 1
            
            for dy, count in diff_table.items():
                if count > max_count:
                    max_count = count
                    best_diffs = (dx, dy)
        
        dap = max_count / self.N
        
        details = {
            'value': float(dap),
            'max_count': int(max_count),
            'best_input_diff': hex(best_diffs[0]),
            'best_output_diff': hex(best_diffs[1]),
            'ideal': 0.015625,
            'score': (0.015625 / dap) * 100 if dap > 0 else 100
        }
        
        return dap, details
    
    def test_differential_uniformity(self) -> Tuple[int, Dict]:
        """
        Test 7: Differential Uniformity (DU)
        
        DU = max δ(a,b) where δ(a,b) = #{x: S(x)⊕S(x⊕a)=b}
        Lower is better, ideal = 4 for AES
        
        Returns:
            (DU_value, details)
        """
        max_du = 0
        
        for a in range(1, self.N):
            for b in range(self.N):
                count = 0
                for x in range(self.N):
                    if (self.sbox[x] ^ self.sbox[x ^ a]) == b:
                        count += 1
                max_du = max(max_du, count)
        
        details = {
            'value': int(max_du),
            'ideal': 4,
            'score': (4 / max_du) * 100 if max_du > 0 else 100
        }
        
        return max_du, details
    
    def test_algebraic_degree(self) -> Tuple[int, Dict]:
        """
        Test 8: Algebraic Degree (AD)
        
        Maximum degree of Boolean functions in ANF representation
        Higher is better, max = 7 for 8-bit S-box
        
        Returns:
            (AD_value, details)
        """
        degrees = []
        
        for bit in range(8):
            # Extract bit function
            f = np.array([(self.sbox[x] >> bit) & 1 for x in range(self.N)])
            
            # Compute ANF using Möbius transform
            anf = f.copy()
            for i in range(self.n):
                for j in range(self.N):
                    if (j >> i) & 1:
                        anf[j] ^= anf[j ^ (1 << i)]
            
            # Find max degree
            max_deg = 0
            for i in range(self.N):
                if anf[i]:
                    deg = bin(i).count('1')
                    max_deg = max(max_deg, deg)
            
            degrees.append(max_deg)
        
        min_degree = min(degrees)
        
        details = {
            'minimum': int(min_degree),
            'per_bit': [int(x) for x in degrees],
            'ideal': 7,
            'score': (min_degree / 7) * 100
        }
        
        return min_degree, details
    
    def test_transparency_order(self) -> Tuple[float, Dict]:
        """
        Test 9: Transparency Order (TO)
        
        Measures resistance to differential power analysis
        Lower is better
        
        Returns:
            (TO_value, details)
        """
        correlations = []
        for bit_in in range(8):
            for bit_out in range(8):
                corr = 0
                for x in range(self.N):
                    in_bit = int((x >> bit_in) & 1)
                    out_bit = int((self.sbox[x] >> bit_out) & 1)
                    corr += (2 * in_bit - 1) * (2 * out_bit - 1)
                correlations.append(abs(corr))
        
        max_corr = max(correlations) / self.N
        
        details = {
            'value': float(max_corr),
            'max_correlation': float(max_corr),
            'score': (1 - max_corr) * 100
        }
        
        return max_corr, details
    
    def test_confusion_index(self) -> Tuple[float, Dict]:
        """
        Test 10: Confusion Index (CI)
        
        Measures output distribution uniformity
        Higher is better
        
        Returns:
            (CI_value, details)
        """
        # Count frequency of each output value
        freq = np.bincount(self.sbox, minlength=256)
        
        # Calculate entropy
        prob = freq / self.N
        prob = prob[prob > 0]  # Remove zeros
        entropy = -np.sum(prob * np.log2(prob))
        
        # Normalize to [0, 1]
        max_entropy = 8.0  # log2(256)
        ci = entropy / max_entropy
        
        # Calculate variance
        expected_freq = self.N / 256
        variance = np.var(freq)
        
        details = {
            'value': float(ci),
            'entropy': float(entropy),
            'max_entropy': float(max_entropy),
            'frequency_variance': float(variance),
            'ideal': 1.0,
            'score': ci * 100
        }
        
        return ci, details
    
    def run_all_tests(self) -> Dict:
        """Run all cryptographic tests"""
        print("Running cryptographic tests...")
        
        results = {}
        
        print("  [1/10] Nonlinearity (NL)...")
        results['NL'] = self.test_nonlinearity()
        
        print("  [2/10] Strict Avalanche Criterion (SAC)...")
        results['SAC'] = self.test_sac()
        
        print("  [3/10] BIC-Nonlinearity (BIC-NL)...")
        results['BIC_NL'] = self.test_bic_nl()
        
        print("  [4/10] BIC-SAC (BIC-SAC) [FIXED]...")
        results['BIC_SAC'] = self.test_bic_sac()
        
        print("  [5/10] Linear Approximation Probability (LAP)...")
        results['LAP'] = self.test_lap()
        
        print("  [6/10] Differential Approximation Probability (DAP)...")
        results['DAP'] = self.test_dap()
        
        print("  [7/10] Differential Uniformity (DU)...")
        results['DU'] = self.test_differential_uniformity()
        
        print("  [8/10] Algebraic Degree (AD)...")
        results['AD'] = self.test_algebraic_degree()
        
        print("  [9/10] Transparency Order (TO)...")
        results['TO'] = self.test_transparency_order()
        
        print("  [10/10] Confusion Index (CI)...")
        results['CI'] = self.test_confusion_index()
        
        print("✓ All tests completed!")
        
        return results


def safe_format(value):
    """Safely format numeric values"""
    if isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, (float, np.floating)):
        return f"{value:.5f}"
    else:
        return str(value)


# Example usage and verification
if __name__ == "__main__":
    from sbox_core_fixed import SBoxConstructor, PredefinedMatrices
    
    print("=" * 70)
    print("Testing S-box_44 (Best S-box from paper)")
    print("=" * 70)
    
    constructor = SBoxConstructor()
    k44_matrix = PredefinedMatrices.get_k44()
    aes_constant = PredefinedMatrices.get_aes_constant()
    sbox44 = constructor.construct_sbox(k44_matrix, aes_constant)
    
    tester = SBoxCryptoTest(sbox44)
    results = tester.run_all_tests()
    
    print("\n" + "=" * 70)
    print("CRYPTOGRAPHIC STRENGTH TEST RESULTS - S-box_44")
    print("=" * 70)
    print(f"{'Test':<20} {'Value':<12} {'Ideal':<12} {'Score'}")
    print("-" * 70)
    
    for test_name, (value, details) in results.items():
        value_str = safe_format(value)
        ideal = details.get('ideal', 'N/A')
        ideal_str = safe_format(ideal) if ideal != 'N/A' else 'N/A'
        score = details.get('score', None)
        score_str = f"{score:.2f}%" if score is not None else "N/A"
        
        print(f"{test_name:<20} {value_str:<12} {ideal_str:<12} {score_str}")
    
    # Compare with paper's expected values for S-box_44
    print("\n" + "=" * 70)
    print("COMPARISON WITH PAPER VALUES (Table 19)")
    print("=" * 70)
    
    paper_values = {
        'NL': 112,
        'SAC': 0.50073,
        'BIC_NL': 112,
        'BIC_SAC': 0.50237,
        'LAP': 0.0625,
        'DAP': 0.01563
    }
    
    print(f"{'Test':<12} {'Paper':<12} {'Computed':<12} {'Match'}")
    print("-" * 70)
    
    for test_name, paper_val in paper_values.items():
        if test_name in results:
            computed_val = results[test_name][0]
            match = abs(computed_val - paper_val) < 0.001
            match_str = "✓" if match else "✗"
            print(f"{test_name:<12} {safe_format(paper_val):<12} {safe_format(computed_val):<12} {match_str}")
    
    print("=" * 70)