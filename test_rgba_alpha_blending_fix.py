#!/usr/bin/env python3

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Test script demonstrating the RGBA to RGB alpha blending fix.

This test verifies that the rgba_to_rgb function now properly implements
alpha blending instead of ignoring the alpha channel.
"""

import sys

import torch

# Add kornia to path for testing
sys.path.insert(0, ".")


def test_rgba_alpha_blending_fix():
    """Test that demonstrates the alpha blending fix."""
    try:
        from kornia.color.rgb import rgba_to_rgb
    except ImportError:
        print("❌ Could not import kornia. Please run from kornia repository root.")
        return False

    print("🧪 Testing RGBA to RGB alpha blending fix...")

    # Test case: Semi-transparent red (50% alpha) on white background
    rgba_red = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0]],  # R channel (red)
            [[0.0, 0.0], [0.0, 0.0]],  # G channel
            [[0.0, 0.0], [0.0, 0.0]],  # B channel
            [[0.5, 0.5], [0.5, 0.5]],
        ]
    )  # A channel (50% transparent)

    print("Input: Semi-transparent red with 50% alpha")
    print("RGBA values: R=1.0, G=0.0, B=0.0, A=0.5")

    # Convert with default white background
    rgb_result = rgba_to_rgb(rgba_red)

    # Expected result with proper alpha blending (white background):
    # R = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    # G = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    # B = 0.5 * 0.0 + 0.5 * 1.0 = 0.5

    r_val = rgb_result[0, 0, 0].item()
    g_val = rgb_result[1, 0, 0].item()
    b_val = rgb_result[2, 0, 0].item()

    print("\nResult with white background:")
    print(f"RGB values: R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")

    # Verify the fix: G and B should be 0.5 (blended with white background)
    # If they were 0.0, it would indicate the old bug (alpha ignored)
    expected_r, expected_g, expected_b = 1.0, 0.5, 0.5
    tolerance = 1e-6

    if (
        abs(r_val - expected_r) < tolerance
        and abs(g_val - expected_g) < tolerance
        and abs(b_val - expected_b) < tolerance
    ):
        print("✅ ALPHA BLENDING WORKS: Colors properly blended with background!")

        # Test with black background to further verify
        rgb_black_bg = rgba_to_rgb(rgba_red, 0.0)
        r_black = rgb_black_bg[0, 0, 0].item()
        g_black = rgb_black_bg[1, 0, 0].item()
        b_black = rgb_black_bg[2, 0, 0].item()

        print("\nResult with black background:")
        print(f"RGB values: R={r_black:.3f}, G={g_black:.3f}, B={b_black:.3f}")

        # With black background: R=0.5, G=0.0, B=0.0
        if abs(r_black - 0.5) < tolerance and abs(g_black - 0.0) < tolerance and abs(b_black - 0.0) < tolerance:
            print("✅ BACKGROUND SUPPORT WORKS: Different backgrounds produce different results!")
            return True
        else:
            print("❌ Background support failed")
            return False
    else:
        print("❌ ALPHA BLENDING FAILED: Alpha channel appears to be ignored!")
        print("Expected: R=1.0, G=0.5, B=0.5")
        print(f"Got:      R={r_val:.3f}, G={g_val:.3f}, B={b_val:.3f}")
        return False


def test_backward_compatibility():
    """Test that existing code still works."""
    try:
        from kornia.color.rgb import rgb_to_rgba, rgba_to_rgb
    except ImportError:
        print("❌ Could not import kornia functions.")
        return False

    print("\n🔄 Testing backward compatibility...")

    # Test lossless conversion with alpha=1.0 (fully opaque)
    original_rgb = torch.ones(3, 4, 4)  # White image
    rgba_opaque = rgb_to_rgba(original_rgb, 1.0)  # Add alpha=1.0
    recovered_rgb = rgba_to_rgb(rgba_opaque)  # Convert back

    if torch.allclose(original_rgb, recovered_rgb, atol=1e-6):
        print("✅ BACKWARD COMPATIBILITY: Lossless conversion with alpha=1.0 works!")
        return True
    else:
        print("❌ Backward compatibility failed!")
        return False


if __name__ == "__main__":
    print("🎃 RGBA to RGB Alpha Blending Fix Test")
    print("=" * 50)

    test1_passed = test_rgba_alpha_blending_fix()
    test2_passed = test_backward_compatibility()

    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The RGBA to RGB alpha blending bug has been successfully fixed!")
    else:
        print("💥 SOME TESTS FAILED!")
        print("The fix may need additional work.")

    print("\nThis fix ensures proper alpha blending in RGBA to RGB conversions,")
    print("resolving the issue where alpha channels were previously ignored.")
