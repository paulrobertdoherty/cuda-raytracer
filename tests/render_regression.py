"""Compare the fixed-seed headless render against the checked-in PPM reference."""

from pathlib import Path
import subprocess
import sys
import tempfile


def main(renderer):
    data = Path(__file__).resolve().parent / "data"
    width, height = 47, 33
    with tempfile.TemporaryDirectory() as work:
        output = Path(work) / "actual.rgba"
        subprocess.run([
            renderer, "--obj", str(data / "tetrahedron.obj"),
            "--width", str(width), "--height", str(height),
            "--samples", "32", "--depth", "8", "--seed", "42", "--fov", "45",
            "--camera-pos", "0,1,2", "--camera-target", "0,0.3,-1",
            "--output", str(output),
        ], check=True)
        actual = output.read_bytes()

    magic, size, maximum, expected = (data / "tetrahedron.ppm").read_bytes().split(b"\n", 3)
    pixels = width * height
    if (magic, size, maximum) != (b"P6", f"{width} {height}".encode(), b"255") or len(expected) != 3 * pixels:
        sys.exit("Invalid reference image")
    if len(actual) != 4 * pixels or actual[3::4] != b"\xff" * pixels:
        sys.exit(f"Expected {pixels} opaque RGBA pixels, received {len(actual)} bytes")

    total_error = 0
    outliers = 0
    for i in range(pixels):
        errors = [abs(actual[4 * i + c] - expected[3 * i + c]) for c in range(3)]
        total_error += sum(errors)
        if max(errors) > 16:
            outliers += 1
    mean_error = total_error / (3 * pixels)
    print(f"Mean absolute RGB error {mean_error:.3f}/255; outlier pixels {outliers}/{pixels}")
    # Allow minor architecture-dependent rounding and occasional divergent sample paths.
    if mean_error > 1 or outliers > 0.01 * pixels:
        sys.exit("Rendered image differs from tetrahedron.ppm")


if __name__ == "__main__":
    main(sys.argv[1])
