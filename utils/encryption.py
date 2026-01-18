"""
Educational Encryption Module
==============================
EDUCATIONAL PROTOTYPE ONLY - NOT FOR REAL SECURITY

This module implements simple XOR encryption with Base64 encoding.
It demonstrates the CONCEPT of data protection, but provides NO real security.

Why XOR encryption is NOT secure:
1. If key is discovered, all data is immediately compromised
2. Vulnerable to known-plaintext attacks
3. Key is hardcoded in the source (never do this in production!)
4. No authentication (can't detect tampering)
5. No initialization vector (patterns are visible)

Real-world alternatives:
- AES-256-GCM (symmetric encryption with authentication)
- RSA-2048+ or ECDSA (asymmetric encryption)
- Proper key derivation: PBKDF2, Argon2
- Secure key storage: Hardware Security Module (HSM), TPM

This implementation is intentionally simple for learning purposes.
"""

import base64
import numpy as np
from config import ENCRYPTION_KEY, ENCRYPTION_ENABLED
import io
from PIL import Image


def encrypt_data(data_bytes: bytes) -> bytes:
    """
    Encrypt raw bytes using XOR cipher + Base64 encoding.
    
    EDUCATIONAL ONLY - This is NOT secure encryption!
    
    How it works:
    1. XOR each byte with corresponding key byte (cycling if needed)
    2. Encode result with Base64 for safe storage as text
    
    Args:
        data_bytes: Raw bytes to encrypt
        
    Returns:
        Base64-encoded encrypted bytes
        
    Example:
        >>> original = b"Hello, World!"
        >>> encrypted = encrypt_data(original)
        >>> decrypted = decrypt_data(encrypted)
        >>> assert original == decrypted
    """
    if not ENCRYPTION_ENABLED:
        # If encryption disabled, just Base64 encode (for demonstration)
        return base64.b64encode(data_bytes)
    
    # XOR each byte with the key (cycling through key bytes)
    # This is the core of XOR cipher - simple but insecure
    xor_bytes = bytes([
        data_byte ^ ENCRYPTION_KEY[i % len(ENCRYPTION_KEY)]
        for i, data_byte in enumerate(data_bytes)
    ])
    
    # Base64 encoding makes binary data safe for text storage
    # (Prevents issues with null bytes, newlines, etc.)
    encrypted_b64 = base64.b64encode(xor_bytes)
    
    return encrypted_b64


def decrypt_data(encrypted_b64: bytes) -> bytes:
    """
    Decrypt Base64-encoded XOR-encrypted data.
    
    EDUCATIONAL ONLY - This is NOT secure decryption!
    
    How it works:
    1. Decode Base64 to get XOR'd bytes
    2. XOR again with same key to reverse the operation
       (XOR property: A ^ B ^ B = A)
    
    Args:
        encrypted_b64: Base64-encoded encrypted bytes
        
    Returns:
        Original decrypted bytes
    """
    if not ENCRYPTION_ENABLED:
        # If encryption disabled, just Base64 decode
        return base64.b64decode(encrypted_b64)
    
    # Decode Base64 to get XOR'd bytes
    xor_bytes = base64.b64decode(encrypted_b64)
    
    # XOR again with the same key to decrypt
    # Mathematical property: (data ^ key) ^ key = data
    decrypted_bytes = bytes([
        xor_byte ^ ENCRYPTION_KEY[i % len(ENCRYPTION_KEY)]
        for i, xor_byte in enumerate(xor_bytes)
    ])
    
    return decrypted_bytes


def encrypt_image(image: Image.Image) -> bytes:
    """
    Encrypt a PIL Image by converting to PNG bytes first.
    
    Args:
        image: PIL Image object
        
    Returns:
        Encrypted bytes (Base64-encoded)
    """
    # Convert image to PNG bytes (lossless format)
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()
    
    # Encrypt the bytes
    return encrypt_data(img_bytes)


def decrypt_image(encrypted_bytes: bytes) -> Image.Image:
    """
    Decrypt encrypted bytes back to a PIL Image.
    
    Args:
        encrypted_bytes: Encrypted image data (Base64-encoded)
        
    Returns:
        PIL Image object
    """
    # Decrypt to get PNG bytes
    img_bytes = decrypt_data(encrypted_bytes)
    
    # Convert bytes back to image
    img_byte_array = io.BytesIO(img_bytes)
    image = Image.open(img_byte_array)
    
    return image


def encrypt_features(feature_vector: np.ndarray) -> bytes:
    """
    Encrypt a numpy feature vector.
    
    Args:
        feature_vector: NumPy array of features (SVM or SqueezeNet)
        
    Returns:
        Encrypted bytes (Base64-encoded)
        
    Note:
        Features are stored as float32 to reduce file size.
        Original precision is maintained during round-trip.
    """
    # Convert numpy array to bytes
    # Using float32 instead of float64 saves 50% space with minimal precision loss
    feature_bytes = feature_vector.astype(np.float32).tobytes()
    
    # Encrypt the bytes
    return encrypt_data(feature_bytes)


def decrypt_features(encrypted_bytes: bytes, feature_dim: int) -> np.ndarray:
    """
    Decrypt encrypted bytes back to a numpy feature vector.
    
    Args:
        encrypted_bytes: Encrypted feature data (Base64-encoded)
        feature_dim: Expected dimensionality (42 for SVM, 512 for SqueezeNet)
        
    Returns:
        NumPy array of features
    """
    # Decrypt to get feature bytes
    feature_bytes = decrypt_data(encrypted_bytes)
    
    # Convert bytes back to numpy array
    feature_vector = np.frombuffer(feature_bytes, dtype=np.float32)
    
    # Validate dimensionality
    if len(feature_vector) != feature_dim:
        raise ValueError(
            f"Decrypted feature dimension mismatch: "
            f"expected {feature_dim}, got {len(feature_vector)}"
        )
    
    return feature_vector


# ============================================================================
# Educational Demonstration Functions
# ============================================================================

def demonstrate_encryption_weakness():
    """
    Educational function to show why XOR encryption is weak.
    
    This function is NOT called by the main system - it's here for learning.
    Run this independently to understand XOR cipher vulnerabilities.
    """
    print("=" * 70)
    print("EDUCATIONAL DEMONSTRATION: XOR Encryption Weaknesses")
    print("=" * 70)
    
    # Example 1: Known plaintext attack
    print("\n1. Known Plaintext Attack:")
    print("   If attacker knows one plaintext-ciphertext pair, they can find the key!")
    
    plaintext = b"KNOWN TEXT"
    encrypted = encrypt_data(plaintext)
    
    print(f"   Plaintext:  {plaintext}")
    print(f"   Encrypted:  {encrypted}")
    
    # Attacker recovers key by XOR'ing plaintext with ciphertext
    ciphertext = base64.b64decode(encrypted)
    recovered_key = bytes([ciphertext[i] ^ plaintext[i] for i in range(len(plaintext))])
    print(f"   Recovered partial key: {recovered_key}")
    print(f"   Actual key (first 10 bytes): {ENCRYPTION_KEY[:10]}")
    
    # Example 2: Pattern visibility
    print("\n2. Pattern Visibility:")
    print("   Identical plaintexts produce identical ciphertexts (no IV)!")
    
    text1 = b"PATTERN"
    text2 = b"PATTERN"
    enc1 = encrypt_data(text1)
    enc2 = encrypt_data(text2)
    
    print(f"   Text 1 encrypted: {enc1}")
    print(f"   Text 2 encrypted: {enc2}")
    print(f"   Are they identical? {enc1 == enc2} ← This is BAD for security!")
    
    print("\n3. What REAL encryption should do:")
    print("   • Different ciphertext each time (use random IV)")
    print("   • Impossible to recover key from plaintext-ciphertext pairs")
    print("   • Authenticated (detect tampering)")
    print("   • Use standards like AES-256-GCM")
    
    print("=" * 70)


if __name__ == "__main__":
    # Run demonstration if this module is executed directly
    demonstrate_encryption_weakness()
    
    # Test round-trip encryption
    print("\nTesting round-trip encryption...")
    
    test_data = b"This is test data for biometric features."
    encrypted = encrypt_data(test_data)
    decrypted = decrypt_data(encrypted)
    
    assert test_data == decrypted, "Round-trip encryption failed!"
    print(f"✓ Original:  {test_data}")
    print(f"✓ Encrypted: {encrypted[:50]}... (truncated)")
    print(f"✓ Decrypted: {decrypted}")
    print("✓ Round-trip successful!")
