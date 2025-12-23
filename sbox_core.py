"""
S-box Construction Module (FIXED VERSION)
Implements core S-box generation based on the paper:
"AES S-box modification uses affine matrices exploration for increased S-box strength"

FIXES:
- Verified affine transformation matches paper exactly
- Corrected bit ordering to match paper examples
"""

import numpy as np
from typing import Tuple, List, Optional

class GF256:
    """Galois Field GF(2^8) operations"""
    
    def __init__(self, poly: int = 0x11B):
        """
        Initialize GF(2^8) with irreducible polynomial
        Default: x^8 + x^4 + x^3 + x + 1 = 0x11B
        """
        self.poly = poly
        self._inv_table = self._generate_inverse_table()
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two elements in GF(2^8)"""
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi_bit = a & 0x80
            a <<= 1
            if hi_bit:
                a ^= self.poly
            b >>= 1
        return p & 0xFF
    
    def power(self, a: int, exp: int) -> int:
        """Calculate a^exp in GF(2^8)"""
        if exp == 0:
            return 1
        result = a
        for _ in range(exp - 1):
            result = self.multiply(result, a)
        return result
    
    def inverse(self, a: int) -> int:
        """Calculate multiplicative inverse in GF(2^8)"""
        if a == 0:
            return 0
        return self._inv_table[a]
    
    def _generate_inverse_table(self) -> np.ndarray:
        """Generate lookup table for multiplicative inverses"""
        inv_table = np.zeros(256, dtype=np.uint8)
        inv_table[0] = 0
        
        for i in range(1, 256):
            for j in range(1, 256):
                if self.multiply(i, j) == 1:
                    inv_table[i] = j
                    break
        return inv_table


class SBoxConstructor:
    """S-box Constructor using affine transformation"""
    
    def __init__(self, irreducible_poly: int = 0x11B):
        self.gf = GF256(irreducible_poly)
        self.inv_matrix = self._generate_inverse_matrix()
    
    def _generate_inverse_matrix(self) -> np.ndarray:
        """Generate 16x16 multiplicative inverse matrix"""
        inv_matrix = np.zeros((16, 16), dtype=np.uint8)
        for i in range(16):
            for j in range(16):
                val = i * 16 + j
                inv_matrix[i, j] = self.gf.inverse(val)
        return inv_matrix
    
    def get_inverse_matrix(self) -> np.ndarray:
        """Get the multiplicative inverse matrix (Table 1 in paper)"""
        return self.inv_matrix.copy()
    
    def affine_transform(self, x: int, affine_matrix: np.ndarray, 
                        constant: np.ndarray) -> int:
        """
        Apply affine transformation: B(x) = K * x^-1 + C (mod 2)
        
        VERIFIED: This matches the paper's Equation (2) and Examples (3) and (4)
        
        Args:
            x: Input value (0-255)
            affine_matrix: 8x8 binary matrix K
            constant: 8x1 binary vector C
        
        Returns:
            Transformed value (0-255)
        """
        # Get inverse
        x_inv = self.gf.inverse(x)
        
        # Convert to binary vector (LSB first: bit 0 at index 0)
        # This matches paper's notation where 199 = 11000111 becomes [1,1,1,0,0,0,1,1]
        x_vec = np.array([(x_inv >> i) & 1 for i in range(8)], dtype=np.uint8)
        
        # Apply affine transformation: K @ x_vec + C (mod 2)
        # K is 8x8 matrix, x_vec is 8x1 column vector
        result = (affine_matrix @ x_vec) % 2
        result = (result + constant.flatten()) % 2
        
        # Convert back to integer (LSB first)
        output = 0
        for i in range(8):
            output |= (result[i] << i)
        
        return output
    
    def construct_sbox(self, affine_matrix: np.ndarray, 
                       constant: np.ndarray) -> np.ndarray:
        """
        Construct complete S-box
        
        Args:
            affine_matrix: 8x8 binary matrix
            constant: 8x1 binary vector
        
        Returns:
            16x16 S-box matrix
        """
        sbox = np.zeros((16, 16), dtype=np.uint8)
        
        for row in range(16):
            for col in range(16):
                input_val = row * 16 + col
                output_val = self.affine_transform(input_val, affine_matrix, constant)
                sbox[row, col] = output_val
        
        return sbox


class PredefinedMatrices:
    """Predefined affine matrices from the paper"""
    
    @staticmethod
    def get_aes_matrix() -> np.ndarray:
        """Original AES affine matrix (K_AES)"""
        return np.array([
            [1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1]
        ], dtype=np.uint8)
    
    @staticmethod
    def get_aes_constant() -> np.ndarray:
        """AES 8-bit constant (C_AES) = 01100011 binary = 0x63"""
        return np.array([[1, 1, 0, 0, 0, 1, 1, 0]], dtype=np.uint8).T
    
    @staticmethod
    def get_k4() -> np.ndarray:
        """Matrix K_4 from paper"""
        return np.array([
            [0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0]
        ], dtype=np.uint8)
    
    @staticmethod
    def get_k44() -> np.ndarray:
        """Matrix K_44 from paper (BEST S-box)"""
        return np.array([
            [0, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 1, 0]
        ], dtype=np.uint8)
    
    @staticmethod
    def get_k81() -> np.ndarray:
        """Matrix K_81 from paper"""
        return np.array([
            [1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1]
        ], dtype=np.uint8)
    
    @staticmethod
    def get_k111() -> np.ndarray:
        """Matrix K_111 from paper"""
        return np.array([
            [1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 1, 1],
            [1, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 0, 1]
        ], dtype=np.uint8)
    
    @staticmethod
    def get_k128() -> np.ndarray:
        """Matrix K_128 from paper"""
        return np.array([
            [1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1]
        ], dtype=np.uint8)
    
    @staticmethod
    def get_all_matrices() -> dict:
        """Get all predefined matrices"""
        return {
            'AES': PredefinedMatrices.get_aes_matrix(),
            'K4': PredefinedMatrices.get_k4(),
            'K44': PredefinedMatrices.get_k44(),
            'K81': PredefinedMatrices.get_k81(),
            'K111': PredefinedMatrices.get_k111(),
            'K128': PredefinedMatrices.get_k128()
        }


# Verification function to match paper's examples
def verify_construction():
    """Verify construction matches paper's examples"""
    print("=" * 70)
    print("VERIFICATION: Checking against paper examples")
    print("=" * 70)
    
    constructor = SBoxConstructor()
    k44_matrix = PredefinedMatrices.get_k44()
    aes_constant = PredefinedMatrices.get_aes_constant()
    
    # Test Example from Equation (3): Input row=0, col=15 (X=15, X^-1=199)
    # Expected output: 214 (decimal) = 11010110 (binary)
    x = 15  # 00001111 binary
    x_inv = 199  # 11000111 binary (from Table 1)
    expected = 214  # 11010110 binary
    
    result = constructor.affine_transform(x, k44_matrix, aes_constant)
    
    print(f"\nExample 1 (Equation 3 from paper):")
    print(f"  Input X = {x} (binary: {bin(x)[2:].zfill(8)})")
    print(f"  X^-1 = {x_inv} (binary: {bin(x_inv)[2:].zfill(8)})")
    print(f"  Expected output: {expected} (binary: {bin(expected)[2:].zfill(8)})")
    print(f"  Computed output: {result} (binary: {bin(result)[2:].zfill(8)})")
    print(f"  ✓ MATCH!" if result == expected else f"  ✗ MISMATCH!")
    
    # Test Example from Equation (4): Input row=15, col=15 (X=255, X^-1=28)
    # Expected output: 70 (decimal) = 01000110 (binary)
    x = 255  # 11111111 binary
    x_inv = 28  # 00011100 binary (from Table 1)
    expected = 70  # 01000110 binary
    
    result = constructor.affine_transform(x, k44_matrix, aes_constant)
    
    print(f"\nExample 2 (Equation 4 from paper):")
    print(f"  Input X = {x} (binary: {bin(x)[2:].zfill(8)})")
    print(f"  X^-1 = {x_inv} (binary: {bin(x_inv)[2:].zfill(8)})")
    print(f"  Expected output: {expected} (binary: {bin(expected)[2:].zfill(8)})")
    print(f"  Computed output: {result} (binary: {bin(result)[2:].zfill(8)})")
    print(f"  ✓ MATCH!" if result == expected else f"  ✗ MISMATCH!")
    
    # Construct full S-box_44 and verify specific values from Table 5
    sbox44 = constructor.construct_sbox(k44_matrix, aes_constant)
    
    print(f"\nS-box_44 verification (Table 5 from paper):")
    print(f"  S-box[0,0] = {sbox44[0,0]} (expected: 99)")
    print(f"  S-box[0,15] = {sbox44[0,15]} (expected: 214)")
    print(f"  S-box[15,15] = {sbox44[15,15]} (expected: 70)")
    
    # Verify a few more random positions from Table 5
    checks = [
        (0, 0, 99), (0, 1, 205), (0, 2, 85), (0, 15, 214),
        (1, 0, 77), (15, 15, 70), (15, 0, 164), (7, 7, 147)
    ]
    
    all_match = True
    for row, col, expected in checks:
        actual = sbox44[row, col]
        match = actual == expected
        all_match = all_match and match
        status = "✓" if match else "✗"
        print(f"  {status} S-box[{row},{col}] = {actual} (expected: {expected})")
    
    print("\n" + "=" * 70)
    print(f"Overall verification: {'✓ ALL TESTS PASSED' if all_match else '✗ SOME TESTS FAILED'}")
    print("=" * 70)
    
    return all_match


# Example usage
if __name__ == "__main__":
    # Run verification
    verify_construction()
    
    print("\n" + "=" * 70)
    print("Constructing all S-boxes from paper...")
    print("=" * 70)
    
    constructor = SBoxConstructor()
    aes_constant = PredefinedMatrices.get_aes_constant()
    matrices = PredefinedMatrices.get_all_matrices()
    
    for name, matrix in matrices.items():
        sbox = constructor.construct_sbox(matrix, aes_constant)
        print(f"\n{name} S-box (first row):")
        print(sbox[0])