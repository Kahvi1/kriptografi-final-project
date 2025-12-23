"""
AES Encryption/Decryption Module with Custom S-box Support
===========================================================

Implements full AES-128 encryption/decryption using custom S-boxes.
Based on the FIPS 197 AES standard with customizable S-box.

Features:
- AES-128 encryption/decryption
- Custom S-box support
- Text and image encryption
- ECB and CBC modes
- PKCS7 padding
"""

import numpy as np
from typing import Tuple, List, Optional
import hashlib

class AESCipher:
    """AES-128 Cipher with Custom S-box Support"""
    
    # AES Constants
    Nb = 4  # Number of columns (32-bit words) in State (always 4 for AES)
    Nk = 4  # Number of 32-bit words in key (4 for AES-128)
    Nr = 10 # Number of rounds (10 for AES-128)
    
    # Rijndael's Galois field
    # GF(2^8) irreducible polynomial: x^8 + x^4 + x^3 + x + 1
    
    def __init__(self, sbox: np.ndarray, key: bytes, mode: str = 'ECB', iv: bytes = None):
        """
        Initialize AES cipher with custom S-box
        
        Args:
            sbox: 16x16 S-box matrix (custom or standard)
            key: 16-byte encryption key for AES-128
            mode: 'ECB' or 'CBC'
            iv: Initialization vector for CBC mode (16 bytes)
        """
        if len(key) != 16:
            raise ValueError("Key must be 16 bytes for AES-128")
        
        self.sbox = sbox.flatten()  # Convert to 1D for fast lookup
        self.inv_sbox = self._generate_inverse_sbox()
        self.mode = mode.upper()
        self.iv = iv if iv else b'\x00' * 16
        
        if self.mode == 'CBC' and len(self.iv) != 16:
            raise ValueError("IV must be 16 bytes for CBC mode")
        
        # Expand key
        self.round_keys = self._key_expansion(key)
    
    def _generate_inverse_sbox(self) -> np.ndarray:
        """Generate inverse S-box for decryption"""
        inv_sbox = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            inv_sbox[self.sbox[i]] = i
        return inv_sbox
    
    # ==================== AES Core Operations ====================
    
    def _sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """SubBytes transformation using custom S-box"""
        return self.sbox[state]
    
    def _inv_sub_bytes(self, state: np.ndarray) -> np.ndarray:
        """Inverse SubBytes transformation"""
        return self.inv_sbox[state]
    
    def _shift_rows(self, state: np.ndarray) -> np.ndarray:
        """ShiftRows transformation"""
        result = state.copy()
        result[1] = np.roll(state[1], -1)
        result[2] = np.roll(state[2], -2)
        result[3] = np.roll(state[3], -3)
        return result
    
    def _inv_shift_rows(self, state: np.ndarray) -> np.ndarray:
        """Inverse ShiftRows transformation"""
        result = state.copy()
        result[1] = np.roll(state[1], 1)
        result[2] = np.roll(state[2], 2)
        result[3] = np.roll(state[3], 3)
        return result
    
    def _xtime(self, a: int) -> int:
        """Multiply by x in GF(2^8)"""
        return ((a << 1) ^ (0x1B if (a & 0x80) else 0)) & 0xFF
    
    def _gf_mult(self, a: int, b: int) -> int:
        """Multiply two numbers in GF(2^8)"""
        p = 0
        for _ in range(8):
            if b & 1:
                p ^= a
            hi_bit = a & 0x80
            a = (a << 1) & 0xFF
            if hi_bit:
                a ^= 0x1B
            b >>= 1
        return p
    
    def _mix_columns(self, state: np.ndarray) -> np.ndarray:
        """MixColumns transformation"""
        result = np.zeros_like(state)
        for c in range(4):
            s0, s1, s2, s3 = state[:, c]
            result[0, c] = self._gf_mult(0x02, s0) ^ self._gf_mult(0x03, s1) ^ s2 ^ s3
            result[1, c] = s0 ^ self._gf_mult(0x02, s1) ^ self._gf_mult(0x03, s2) ^ s3
            result[2, c] = s0 ^ s1 ^ self._gf_mult(0x02, s2) ^ self._gf_mult(0x03, s3)
            result[3, c] = self._gf_mult(0x03, s0) ^ s1 ^ s2 ^ self._gf_mult(0x02, s3)
        return result
    
    def _inv_mix_columns(self, state: np.ndarray) -> np.ndarray:
        """Inverse MixColumns transformation"""
        result = np.zeros_like(state)
        for c in range(4):
            s0, s1, s2, s3 = state[:, c]
            result[0, c] = self._gf_mult(0x0e, s0) ^ self._gf_mult(0x0b, s1) ^ self._gf_mult(0x0d, s2) ^ self._gf_mult(0x09, s3)
            result[1, c] = self._gf_mult(0x09, s0) ^ self._gf_mult(0x0e, s1) ^ self._gf_mult(0x0b, s2) ^ self._gf_mult(0x0d, s3)
            result[2, c] = self._gf_mult(0x0d, s0) ^ self._gf_mult(0x09, s1) ^ self._gf_mult(0x0e, s2) ^ self._gf_mult(0x0b, s3)
            result[3, c] = self._gf_mult(0x0b, s0) ^ self._gf_mult(0x0d, s1) ^ self._gf_mult(0x09, s2) ^ self._gf_mult(0x0e, s3)
        return result
    
    def _add_round_key(self, state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
        """AddRoundKey transformation"""
        return state ^ round_key
    
    # ==================== Key Expansion ====================
    
    def _rot_word(self, word: np.ndarray) -> np.ndarray:
        """Rotate word left by one byte"""
        return np.roll(word, -1)
    
    def _sub_word(self, word: np.ndarray) -> np.ndarray:
        """Apply S-box to each byte in word"""
        return self.sbox[word]
    
    def _key_expansion(self, key: bytes) -> List[np.ndarray]:
        """Expand 128-bit key into 11 round keys"""
        # Rcon: Round constants
        Rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]
        
        # Convert key to 4x4 matrix
        key_array = np.frombuffer(key, dtype=np.uint8).reshape(4, 4, order='F')
        
        # Initialize with original key
        w = [key_array[:, i].copy() for i in range(4)]
        
        # Generate remaining words
        for i in range(4, self.Nb * (self.Nr + 1)):
            temp = w[i - 1].copy()
            
            if i % self.Nk == 0:
                temp = self._sub_word(self._rot_word(temp))
                temp[0] ^= Rcon[i // self.Nk - 1]
            
            w.append(w[i - self.Nk] ^ temp)
        
        # Group into round keys
        round_keys = []
        for r in range(self.Nr + 1):
            round_key = np.column_stack([w[r * 4 + i] for i in range(4)])
            round_keys.append(round_key)
        
        return round_keys
    
    # ==================== Encryption/Decryption ====================
    
    def _encrypt_block(self, plaintext: np.ndarray) -> np.ndarray:
        """Encrypt single 16-byte block"""
        # Convert to state matrix (column-major)
        state = plaintext.reshape(4, 4, order='F')
        
        # Initial round
        state = self._add_round_key(state, self.round_keys[0])
        
        # Main rounds
        for round in range(1, self.Nr):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, self.round_keys[round])
        
        # Final round (no MixColumns)
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, self.round_keys[self.Nr])
        
        return state.flatten(order='F')
    
    def _decrypt_block(self, ciphertext: np.ndarray) -> np.ndarray:
        """Decrypt single 16-byte block"""
        # Convert to state matrix (column-major)
        state = ciphertext.reshape(4, 4, order='F')
        
        # Initial round
        state = self._add_round_key(state, self.round_keys[self.Nr])
        
        # Main rounds
        for round in range(self.Nr - 1, 0, -1):
            state = self._inv_shift_rows(state)
            state = self._inv_sub_bytes(state)
            state = self._add_round_key(state, self.round_keys[round])
            state = self._inv_mix_columns(state)
        
        # Final round (no InvMixColumns)
        state = self._inv_shift_rows(state)
        state = self._inv_sub_bytes(state)
        state = self._add_round_key(state, self.round_keys[0])
        
        return state.flatten(order='F')
    
    # ==================== Padding ====================
    
    def _pad(self, data: bytes) -> bytes:
        """Apply PKCS7 padding"""
        pad_len = 16 - (len(data) % 16)
        return data + bytes([pad_len] * pad_len)
    
    def _unpad(self, data: bytes) -> bytes:
        """Remove PKCS7 padding"""
        pad_len = data[-1]
        return data[:-pad_len]
    
    # ==================== Public API ====================
    
    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt plaintext
        
        Args:
            plaintext: Data to encrypt
        
        Returns:
            Encrypted ciphertext
        """
        # Pad plaintext
        padded = self._pad(plaintext)
        
        # Split into blocks
        blocks = [np.frombuffer(padded[i:i+16], dtype=np.uint8) 
                  for i in range(0, len(padded), 16)]
        
        encrypted_blocks = []
        
        if self.mode == 'ECB':
            for block in blocks:
                encrypted_blocks.append(self._encrypt_block(block))
        
        elif self.mode == 'CBC':
            prev_block = np.frombuffer(self.iv, dtype=np.uint8)
            for block in blocks:
                # XOR with previous ciphertext block
                block = block ^ prev_block
                encrypted = self._encrypt_block(block)
                encrypted_blocks.append(encrypted)
                prev_block = encrypted
        
        return b''.join([block.tobytes() for block in encrypted_blocks])
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt ciphertext
        
        Args:
            ciphertext: Data to decrypt
        
        Returns:
            Decrypted plaintext
        """
        if len(ciphertext) % 16 != 0:
            raise ValueError("Ciphertext length must be multiple of 16")
        
        # Split into blocks
        blocks = [np.frombuffer(ciphertext[i:i+16], dtype=np.uint8) 
                  for i in range(0, len(ciphertext), 16)]
        
        decrypted_blocks = []
        
        if self.mode == 'ECB':
            for block in blocks:
                decrypted_blocks.append(self._decrypt_block(block))
        
        elif self.mode == 'CBC':
            prev_block = np.frombuffer(self.iv, dtype=np.uint8)
            for block in blocks:
                decrypted = self._decrypt_block(block)
                # XOR with previous ciphertext block
                decrypted = decrypted ^ prev_block
                decrypted_blocks.append(decrypted)
                prev_block = block
        
        # Join and unpad
        plaintext = b''.join([block.tobytes() for block in decrypted_blocks])
        return self._unpad(plaintext)


class AESImageCipher:
    """AES Cipher for Image Encryption"""
    
    def __init__(self, sbox: np.ndarray, key: bytes, mode: str = 'ECB'):
        """Initialize image cipher"""
        self.cipher = AESCipher(sbox, key, mode)
    
    def encrypt_image(self, image_array: np.ndarray) -> Tuple[bytes, dict]:
        """
        Encrypt image array
        
        Args:
            image_array: Image as numpy array (H, W, C) or (H, W)
        
        Returns:
            (encrypted_bytes, metadata)
        """
        # Save original shape and dtype
        original_shape = image_array.shape
        original_dtype = image_array.dtype
        
        # Flatten and convert to bytes
        flat_data = image_array.flatten().astype(np.uint8)
        data_bytes = flat_data.tobytes()
        
        # Encrypt (this will add padding)
        encrypted_bytes = self.cipher.encrypt(data_bytes)
        
        metadata = {
            'original_shape': original_shape,
            'original_dtype': str(original_dtype),
            'data_length': len(data_bytes),
            'encrypted_length': len(encrypted_bytes)
        }
        
        return encrypted_bytes, metadata
    
    def decrypt_image(self, encrypted_bytes: bytes, 
                     metadata: dict) -> np.ndarray:
        """
        Decrypt image
        
        Args:
            encrypted_bytes: Encrypted data
            metadata: Metadata from encryption
        
        Returns:
            Decrypted image array
        """
        # Decrypt
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        
        # Convert back to array
        decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        
        # Reshape to original
        decrypted_array = decrypted_array.reshape(metadata['original_shape'])
        
        return decrypted_array


# ==================== Utility Functions ====================

def generate_key_from_password(password: str) -> bytes:
    """Generate 16-byte key from password using SHA-256"""
    return hashlib.sha256(password.encode()).digest()[:16]


def generate_random_key() -> bytes:
    """Generate random 16-byte key"""
    return np.random.bytes(16)


def generate_random_iv() -> bytes:
    """Generate random 16-byte IV for CBC mode"""
    return np.random.bytes(16)


# ==================== Demo Functions ====================

def demo_text_encryption():
    """Demo text encryption with custom S-box"""
    from sbox_core import SBoxConstructor, PredefinedMatrices
    
    print("="*70)
    print("AES TEXT ENCRYPTION DEMO")
    print("="*70)
    
    # Create S-box_44 (best from paper)
    constructor = SBoxConstructor()
    k44 = PredefinedMatrices.get_k44()
    constant = PredefinedMatrices.get_aes_constant()
    sbox44 = constructor.construct_sbox(k44, constant)
    
    # Generate key from password
    password = "MySecretPassword123"
    key = generate_key_from_password(password)
    
    # Create cipher
    cipher = AESCipher(sbox44, key, mode='ECB')
    
    # Test message
    message = "Hello, this is a secret message encrypted with custom S-box_44!"
    print(f"\nOriginal:  {message}")
    
    # Encrypt
    ciphertext = cipher.encrypt(message.encode())
    print(f"Encrypted: {ciphertext.hex()}")
    
    # Decrypt
    decrypted = cipher.decrypt(ciphertext)
    print(f"Decrypted: {decrypted.decode()}")
    
    # Verify
    print(f"\nVerification: {'✓ SUCCESS' if decrypted.decode() == message else '✗ FAILED'}")


if __name__ == "__main__":
    demo_text_encryption()