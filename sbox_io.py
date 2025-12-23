"""
S-box File I/O Module
Handles import/export of S-boxes in various formats (Excel, CSV, TXT)
"""

import numpy as np
import pandas as pd
import io
from typing import Tuple, Optional

class SBoxIO:
    """S-box file input/output handler"""
    
    @staticmethod
    def sbox_to_dataframe(sbox: np.ndarray, format_type: str = 'decimal') -> pd.DataFrame:
        """
        Convert S-box to pandas DataFrame
        
        Args:
            sbox: 16x16 S-box matrix
            format_type: 'decimal' or 'binary' or 'hex'
        
        Returns:
            DataFrame representation
        """
        if format_type == 'binary':
            # Convert to 8-bit binary strings
            data = [[f'{val:08b}' for val in row] for row in sbox]
        elif format_type == 'hex':
            # Convert to hexadecimal
            data = [[f'{val:02X}' for val in row] for row in sbox]
        else:
            # Keep as decimal
            data = sbox.tolist()
        
        # Create DataFrame with row/column labels
        df = pd.DataFrame(
            data,
            index=[f'Row {i}' for i in range(16)],
            columns=[f'Col {i}' for i in range(16)]
        )
        
        return df
    
    @staticmethod
    def export_to_excel(sbox: np.ndarray, filename: str, 
                       include_formats: list = ['decimal']) -> bytes:
        """
        Export S-box to Excel file
        
        Args:
            sbox: 16x16 S-box matrix
            filename: Output filename
            include_formats: List of formats to include ['decimal', 'binary', 'hex']
        
        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for fmt in include_formats:
                df = SBoxIO.sbox_to_dataframe(sbox, fmt)
                sheet_name = fmt.capitalize()
                df.to_excel(writer, sheet_name=sheet_name)
        
        output.seek(0)
        return output.getvalue()
    
    @staticmethod
    def export_to_csv(sbox: np.ndarray, format_type: str = 'decimal') -> str:
        """
        Export S-box to CSV string
        
        Args:
            sbox: 16x16 S-box matrix
            format_type: 'decimal', 'binary', or 'hex'
        
        Returns:
            CSV string
        """
        df = SBoxIO.sbox_to_dataframe(sbox, format_type)
        return df.to_csv()
    
    @staticmethod
    def export_to_txt(sbox: np.ndarray, format_type: str = 'decimal') -> str:
        """
        Export S-box to formatted text
        
        Args:
            sbox: 16x16 S-box matrix
            format_type: 'decimal', 'binary', or 'hex'
        
        Returns:
            Formatted text string
        """
        output = []
        output.append("=" * 80)
        output.append(f"S-BOX ({format_type.upper()} FORMAT)")
        output.append("=" * 80)
        output.append("")
        
        # Column headers
        if format_type == 'binary':
            output.append("     " + "  ".join([f"{i:2d}" for i in range(16)]))
            output.append("     " + "  ".join(["--------" for _ in range(16)]))
        else:
            output.append("    " + " ".join([f"{i:3d}" for i in range(16)]))
            output.append("    " + " ".join(["---" for _ in range(16)]))
        
        # Data rows
        for i, row in enumerate(sbox):
            if format_type == 'binary':
                values = "  ".join([f"{val:08b}" for val in row])
                output.append(f"{i:2d}  {values}")
            elif format_type == 'hex':
                values = " ".join([f"{val:3X}" for val in row])
                output.append(f"{i:2d}  {values}")
            else:
                values = " ".join([f"{val:3d}" for val in row])
                output.append(f"{i:2d}  {values}")
        
        output.append("")
        output.append("=" * 80)
        
        return "\n".join(output)
    
    @staticmethod
    def import_from_csv(file_content: str) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Import S-box from CSV
        
        Args:
            file_content: CSV file content as string
        
        Returns:
            (success, sbox_matrix, error_message)
        """
        try:
            # Try to read CSV
            df = pd.read_csv(io.StringIO(file_content), index_col=0)
            
            # Check dimensions
            if df.shape != (16, 16):
                return False, None, f"Invalid dimensions: {df.shape}. Expected (16, 16)"
            
            # Convert to numpy array
            sbox = df.values.astype(int)
            
            # Validate range
            if np.any(sbox < 0) or np.any(sbox > 255):
                return False, None, "Values must be in range [0, 255]"
            
            return True, sbox, "Success"
            
        except Exception as e:
            return False, None, f"Error parsing CSV: {str(e)}"
    
    @staticmethod
    def import_from_excel(file_content: bytes, 
                         sheet_name: str = 'Decimal') -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Import S-box from Excel
        
        Args:
            file_content: Excel file content as bytes
            sheet_name: Sheet name to read
        
        Returns:
            (success, sbox_matrix, error_message)
        """
        try:
            # Read Excel
            df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name, index_col=0)
            
            # Check dimensions
            if df.shape != (16, 16):
                return False, None, f"Invalid dimensions: {df.shape}. Expected (16, 16)"
            
            # Convert to numpy array
            sbox = df.values.astype(int)
            
            # Validate range
            if np.any(sbox < 0) or np.any(sbox > 255):
                return False, None, "Values must be in range [0, 255]"
            
            return True, sbox, "Success"
            
        except Exception as e:
            return False, None, f"Error parsing Excel: {str(e)}"
    
    @staticmethod
    def import_from_txt(file_content: str) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Import S-box from formatted text
        
        Args:
            file_content: Text file content
        
        Returns:
            (success, sbox_matrix, error_message)
        """
        try:
            lines = file_content.strip().split('\n')
            
            # Find data lines (skip headers and separators)
            data_lines = []
            for line in lines:
                # Skip empty lines and separator lines
                if not line.strip() or '=' in line or 'S-BOX' in line:
                    continue
                
                # Skip header line with column numbers
                if 'Col' in line or all(c.isdigit() or c.isspace() or c == '-' for c in line):
                    continue
                
                # Extract data
                parts = line.split()
                if len(parts) >= 16:
                    # First part might be row number, take last 16 values
                    row_data = parts[-16:]
                    
                    # Handle binary format (8 digits)
                    if all(len(x) == 8 and all(c in '01' for c in x) for x in row_data):
                        row_data = [int(x, 2) for x in row_data]
                    # Handle hex format
                    elif all(len(x) <= 3 for x in row_data):
                        try:
                            row_data = [int(x, 16) for x in row_data]
                        except:
                            row_data = [int(x) for x in row_data]
                    else:
                        row_data = [int(x) for x in row_data]
                    
                    data_lines.append(row_data)
            
            # Check if we have 16 rows
            if len(data_lines) != 16:
                return False, None, f"Expected 16 rows, found {len(data_lines)}"
            
            # Convert to numpy array
            sbox = np.array(data_lines, dtype=np.uint8)
            
            # Validate range
            if np.any(sbox < 0) or np.any(sbox > 255):
                return False, None, "Values must be in range [0, 255]"
            
            return True, sbox, "Success"
            
        except Exception as e:
            return False, None, f"Error parsing text file: {str(e)}"
    
    @staticmethod
    def export_test_results(sbox_name: str, test_results: dict, 
                           format_type: str = 'excel') -> bytes:
        """
        Export cryptographic test results
        
        Args:
            sbox_name: Name of the S-box
            test_results: Dictionary of test results
            format_type: 'excel' or 'csv'
        
        Returns:
            File content as bytes
        """
        # Create summary DataFrame
        summary_data = []
        for test_name, (value, details) in test_results.items():
            summary_data.append({
                'Test': test_name,
                'Value': value,
                'Ideal': details.get('ideal', 'N/A'),
                'Score': f"{details.get('score', 0):.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        if format_type == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed sheets for tests with matrices
                if 'SAC' in test_results:
                    sac_matrix = pd.DataFrame(
                        test_results['SAC'][1]['matrix'],
                        index=[f'Input {i}' for i in range(8)],
                        columns=[f'Output {i}' for i in range(8)]
                    )
                    sac_matrix.to_excel(writer, sheet_name='SAC_Matrix')
                
                if 'BIC_SAC' in test_results:
                    bic_sac_matrix = pd.DataFrame(
                        test_results['BIC_SAC'][1]['matrix'],
                        index=[f'Bit {i}' for i in range(8)],
                        columns=[f'Bit {i}' for i in range(8)]
                    )
                    bic_sac_matrix.to_excel(writer, sheet_name='BIC_SAC_Matrix')
            
            output.seek(0)
            return output.getvalue()
        
        else:  # CSV
            return summary_df.to_csv(index=False).encode()


# Example usage
if __name__ == "__main__":
    from sbox_core import SBoxConstructor, PredefinedMatrices
    
    # Create S-box_44
    constructor = SBoxConstructor()
    k44_matrix = PredefinedMatrices.get_k44()
    aes_constant = PredefinedMatrices.get_aes_constant()
    sbox44 = constructor.construct_sbox(k44_matrix, aes_constant)
    
    # Export to different formats
    print("Exporting S-box_44...")
    
    # TXT format
    txt_output = SBoxIO.export_to_txt(sbox44, 'decimal')
    print(txt_output[:500])  # Print first 500 chars
    
    # CSV format
    csv_output = SBoxIO.export_to_csv(sbox44)
    print(f"\nCSV output length: {len(csv_output)} bytes")
    
    # Excel format
    excel_output = SBoxIO.export_to_excel(sbox44, 'sbox44.xlsx', 
                                          include_formats=['decimal', 'binary', 'hex'])
    print(f"Excel output length: {len(excel_output)} bytes")