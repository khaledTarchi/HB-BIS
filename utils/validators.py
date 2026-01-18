"""
Input Validation Module
========================
Educational Prototype - HB-BIS

This module provides validation functions to ensure data quality and system stability.
Validation is crucial in biometric systems to prevent errors and security issues.

Key validation aspects:
1. Image format validation (prevent malicious files)
2. Image quality validation (ensure usable biometric samples)
3. Username validation (prevent injection attacks, file system issues)
4. Resource validation (prevent disk space exhaustion)
"""

import os
import re
from PIL import Image
import numpy as np
import shutil
from typing import Tuple, Optional
from config import (
    SUPPORTED_IMAGE_FORMATS,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    MAX_IMAGE_HEIGHT,
    MIN_CONTRAST_RATIO
)


def validate_image_format(filepath: str) -> Tuple[bool, str]:
    """
    Validate that the file is a supported image format.
    
    This performs two checks:
    1. Extension check (quick, but can be spoofed)
    2. Magic byte check (reads file header, more reliable)
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Educational Note:
        Never trust file extensions alone! Attackers can rename malicious files.
        Always check the actual file content (magic bytes/file signature).
    """
    # Check if file exists
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"
    
    # Check file extension (quick pre-filter)
    file_ext = os.path.splitext(filepath)[1].lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        return False, f"Unsupported format: {file_ext}. Supported: {SUPPORTED_IMAGE_FORMATS}"
    
    # Check magic bytes by trying to open with PIL
    # PIL validates the file header, not just the extension
    try:
        with Image.open(filepath) as img:
            # Force loading to ensure it's a valid image
            img.verify()
        return True, "Valid image format"
    
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def validate_image_quality(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate that the image meets minimum quality requirements for biometrics.
    
    Checks:
    1. Minimum resolution (too small = insufficient detail)
    2. Maximum resolution (too large = processing issues)
    3. Contrast ratio (too low = likely blank/damaged image)
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Educational Note:
        Biometric systems need minimum quality standards. Low-quality samples
        lead to poor matching accuracy and high false rejection rates.
    """
    width, height = image.size
    
    # Check minimum dimensions
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return False, (
            f"Image too small: {width}x{height}. "
            f"Minimum required: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}"
        )
    
    # Check maximum dimensions (prevent memory issues)
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        return False, (
            f"Image too large: {width}x{height}. "
            f"Maximum allowed: {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"
        )
    
    # Check contrast ratio (detect blank or severely damaged images)
    # Convert to grayscale for analysis
    gray_image = image.convert('L')
    pixels = np.array(gray_image)
    
    # Calculate contrast as (max - min) / max
    # This is a simple measure; professional systems use more sophisticated metrics
    min_intensity = pixels.min()
    max_intensity = pixels.max()
    
    if max_intensity == 0:
        return False, "Image is completely black (no content)"
    
    contrast = (max_intensity - min_intensity) / max_intensity
    
    if contrast < MIN_CONTRAST_RATIO:
        return False, (
            f"Insufficient contrast: {contrast:.2f}. "
            f"Image may be blank or severely faded. Minimum: {MIN_CONTRAST_RATIO}"
        )
    
    return True, "Image quality acceptable"


def validate_user_name(name: str) -> Tuple[bool, str]:
    """
    Validate and sanitize username for safe storage.
    
    Security checks:
    1. Non-empty
    2. Reasonable length
    3. No path traversal characters (../, etc.)
    4. No special characters that could cause file system issues
    
    Args:
        name: Proposed username
        
    Returns:
        Tuple of (is_valid, error_message or sanitized_name)
        
    Educational Note:
        Input validation prevents:
        - Path traversal attacks (accessing files outside intended directory)
        - SQL injection (if using SQL database)
        - File system errors (invalid characters)
        - Denial of service (excessively long names)
    """
    # Check not empty
    if not name or name.strip() == "":
        return False, "Username cannot be empty"
    
    # Remove leading/trailing whitespace
    name = name.strip()
    
    # Check length
    if len(name) < 2:
        return False, "Username must be at least 2 characters"
    
    if len(name) > 50:
        return False, "Username too long (maximum 50 characters)"
    
    # Check for path traversal attempts
    # These are common attack patterns to access files outside intended directory
    dangerous_patterns = ['../', '..\\', '/', '\\', '\x00']
    for pattern in dangerous_patterns:
        if pattern in name:
            return False, f"Username contains invalid characters: {pattern}"
    
    # Check for valid characters (alphanumeric, spaces, hyphens, underscores)
    # This regex prevents most file system and injection issues
    if not re.match(r'^[a-zA-Z0-9 _-]+$', name):
        return False, (
            "Username can only contain letters, numbers, spaces, "
            "hyphens, and underscores"
        )
    
    # Sanitize: replace spaces with underscores for file system compatibility
    sanitized_name = name.replace(' ', '_')
    
    return True, sanitized_name


def is_disk_space_available(required_mb: float = 10) -> Tuple[bool, str]:
    """
    Check if sufficient disk space is available.
    
    Args:
        required_mb: Required free space in megabytes
        
    Returns:
        Tuple of (is_available, message)
        
    Educational Note:
        Biometric databases can grow large. Always check available space
        before storing new samples to prevent partial writes and corruption.
    """
    try:
        # Get disk usage statistics for current directory
        stat = shutil.disk_usage(os.getcwd())
        
        # Convert to megabytes
        free_mb = stat.free / (1024 * 1024)
        
        if free_mb < required_mb:
            return False, (
                f"Insufficient disk space: {free_mb:.1f} MB available, "
                f"{required_mb:.1f} MB required"
            )
        
        return True, f"Sufficient space available: {free_mb:.1f} MB free"
    
    except Exception as e:
        # If we can't check disk space, be conservative and return warning
        return False, f"Could not check disk space: {str(e)}"


def validate_feature_vector(features: np.ndarray, expected_dim: int) -> Tuple[bool, str]:
    """
    Validate that a feature vector has the correct dimensionality and valid values.
    
    Args:
        features: NumPy array of features
        expected_dim: Expected dimensionality (42 for SVM, 512 for SqueezeNet)
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Educational Note:
        Feature vectors must have consistent dimensionality for comparison.
        Invalid values (NaN, Inf) indicate feature extraction failures.
    """
    # Check dimensionality
    if features.shape[0] != expected_dim:
        return False, (
            f"Feature dimension mismatch: expected {expected_dim}, "
            f"got {features.shape[0]}"
        )
    
    # Check for invalid values (NaN or Inf)
    if np.any(np.isnan(features)):
        return False, "Feature vector contains NaN values (feature extraction error)"
    
    if np.any(np.isinf(features)):
        return False, "Feature vector contains infinite values (normalization error)"
    
    return True, "Feature vector valid"


# ============================================================================
# Educational Demonstration
# ============================================================================

if __name__ == "__main__":
    """
    Demonstrate validation functions with example inputs.
    Run this module directly to see validation in action.
    """
    print("=" * 70)
    print("EDUCATIONAL DEMONSTRATION: Input Validation")
    print("=" * 70)
    
    # Test username validation
    print("\n1. Username Validation:")
    test_names = [
        "John Doe",          # Valid
        "user_123",          # Valid
        "a",                 # Too short
        "../etc/passwd",     # Path traversal attempt!
        "user@email.com",    # Special char (@)
        "Valid-User-Name",   # Valid
    ]
    
    for name in test_names:
        is_valid, result = validate_user_name(name)
        status = "✓" if is_valid else "✗"
        print(f"   {status} '{name}' → {result}")
    
    # Test disk space
    print("\n2. Disk Space Check:")
    is_available, message = is_disk_space_available(required_mb=10)
    print(f"   {'✓' if is_available else '✗'} {message}")
    
    # Test feature validation
    print("\n3. Feature Vector Validation:")
    
    valid_features = np.random.randn(42)
    is_valid, msg = validate_feature_vector(valid_features, expected_dim=42)
    print(f"   ✓ Valid SVM features (42D): {msg}")
    
    invalid_features = np.random.randn(40)  # Wrong dimension
    is_valid, msg = validate_feature_vector(invalid_features, expected_dim=42)
    print(f"   ✗ Invalid features (40D): {msg}")
    
    nan_features = np.full(42, np.nan)  # NaN values
    is_valid, msg = validate_feature_vector(nan_features, expected_dim=42)
    print(f"   ✗ NaN features: {msg}")
    
    print("\n" + "=" * 70)
    print("Validation is critical for system reliability and security!")
    print("=" * 70)
