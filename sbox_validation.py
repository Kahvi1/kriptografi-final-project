"""
S-box Validation Module
Implements balance and bijective tests as described in the paper
"""

import numpy as np
from typing import Tuple, Dict

class SBoxValidator:
    """Validator for S-box balance and bijectivity criteria"""
    
    @staticmethod
    def balance_test(sbox: np.ndarray) -> Tuple[bool, Dict]:
        """
        Test if S-box satisfies balance criterion
        
        Balance: #{x|f(x)=0} = #{x|f(x)=1} for each bit position
        For GF(2^8), each side should equal 128
        
        Args:
            sbox: 16x16 S-box matrix
        
        Returns:
            (is_balanced, detailed_results)
        """
        # Flatten S-box to 1D array
        sbox_flat = sbox.flatten()
        
        # Convert each value to 8-bit binary
        results = {}
        all_balanced = True
        
        for bit_pos in range(8):
            # Count zeros and ones at this bit position
            zeros = 0
            ones = 0
            
            for val in sbox_flat:
                bit = (val >> bit_pos) & 1
                if bit == 0:
                    zeros += 1
                else:
                    ones += 1
            
            is_balanced = (zeros == 128 and ones == 128)
            results[f'f{bit_pos}'] = {
                'zeros': zeros,
                'ones': ones,
                'balanced': is_balanced
            }
            
            if not is_balanced:
                all_balanced = False
        
        return all_balanced, results
    
    @staticmethod
    def bijective_test(sbox: np.ndarray) -> Tuple[bool, Dict]:
        """
        Test if S-box satisfies bijective criterion
        
        Bijective: Each output value [0, 255] appears exactly once
        
        Args:
            sbox: 16x16 S-box matrix
        
        Returns:
            (is_bijective, detailed_results)
        """
        # Flatten S-box
        sbox_flat = sbox.flatten()
        
        # Check uniqueness
        unique_values = np.unique(sbox_flat)
        
        # Check range [0, 255]
        in_range = np.all((sbox_flat >= 0) & (sbox_flat <= 255))
        
        # Check if all values appear exactly once
        is_bijective = (len(unique_values) == 256) and in_range
        
        # Find duplicates or missing values
        all_values = set(range(256))
        present_values = set(sbox_flat)
        missing_values = list(all_values - present_values)
        
        # Find duplicates
        value_counts = {}
        for val in sbox_flat:
            value_counts[val] = value_counts.get(val, 0) + 1
        
        duplicates = {val: count for val, count in value_counts.items() if count > 1}
        
        results = {
            'unique_count': len(unique_values),
            'expected_count': 256,
            'in_range': in_range,
            'is_bijective': is_bijective,
            'missing_values': missing_values[:10] if len(missing_values) <= 10 else f"{len(missing_values)} values",
            'duplicates': duplicates if len(duplicates) <= 10 else f"{len(duplicates)} duplicates"
        }
        
        return is_bijective, results
    
    @staticmethod
    def validate_sbox(sbox: np.ndarray) -> Tuple[bool, Dict]:
        """
        Complete validation: balance + bijective
        
        Args:
            sbox: 16x16 S-box matrix
        
        Returns:
            (is_valid, all_results)
        """
        # Balance test
        is_balanced, balance_results = SBoxValidator.balance_test(sbox)
        
        # Bijective test
        is_bijective, bijective_results = SBoxValidator.bijective_test(sbox)
        
        # Overall validity
        is_valid = is_balanced and is_bijective
        
        all_results = {
            'valid': is_valid,
            'balanced': is_balanced,
            'bijective': is_bijective,
            'balance_details': balance_results,
            'bijective_details': bijective_results
        }
        
        return is_valid, all_results
    
    @staticmethod
    def print_validation_report(results: Dict) -> None:
        """Print formatted validation report"""
        print("=" * 60)
        print("S-BOX VALIDATION REPORT")
        print("=" * 60)
        
        # Overall status
        status = "✓ VALID" if results['valid'] else "✗ INVALID"
        print(f"\nOverall Status: {status}")
        print(f"  - Balance Test: {'✓ PASS' if results['balanced'] else '✗ FAIL'}")
        print(f"  - Bijective Test: {'✓ PASS' if results['bijective'] else '✗ FAIL'}")
        
        # Balance details
        print("\n" + "-" * 60)
        print("BALANCE TEST DETAILS")
        print("-" * 60)
        print(f"{'Bit':>5} | {'Zeros':>6} | {'Ones':>6} | {'Status':>10}")
        print("-" * 60)
        
        for bit_pos in range(8):
            detail = results['balance_details'][f'f{bit_pos}']
            status = "✓ PASS" if detail['balanced'] else "✗ FAIL"
            print(f"  f{bit_pos}  | {detail['zeros']:>6} | {detail['ones']:>6} | {status:>10}")
        
        # Bijective details
        print("\n" + "-" * 60)
        print("BIJECTIVE TEST DETAILS")
        print("-" * 60)
        bij_details = results['bijective_details']
        print(f"Unique values: {bij_details['unique_count']}/256")
        print(f"In range [0,255]: {'Yes' if bij_details['in_range'] else 'No'}")
        
        if not bij_details['is_bijective']:
            print(f"\nMissing values: {bij_details['missing_values']}")
            print(f"Duplicate values: {bij_details['duplicates']}")
        
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    from sbox_core import SBoxConstructor, PredefinedMatrices
    
    # Test AES S-box
    constructor = SBoxConstructor()
    aes_matrix = PredefinedMatrices.get_aes_matrix()
    aes_constant = PredefinedMatrices.get_aes_constant()
    aes_sbox = constructor.construct_sbox(aes_matrix, aes_constant)
    
    print("Testing AES S-box:")
    is_valid, results = SBoxValidator.validate_sbox(aes_sbox)
    SBoxValidator.print_validation_report(results)
    
    # Test S-box_44
    k44_matrix = PredefinedMatrices.get_k44()
    sbox44 = constructor.construct_sbox(k44_matrix, aes_constant)
    
    print("\n\nTesting S-box_44:")
    is_valid, results = SBoxValidator.validate_sbox(sbox44)
    SBoxValidator.print_validation_report(results)