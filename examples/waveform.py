"""A pulse from your own I/Q arrays: a frequency-shifted Gaussian 90°.

``gaussian_90`` stands in for any external generator (an optimiser, a shape tool): it
returns I and Q normalised to 1 and the parameters nutate needs, the duration and the Rabi
frequency. ``nt.waveform`` turns them into a pulse and ``nt.run`` writes the video, a still
and the B1 figures. Illustrative parameters: 200 µs, 512 points, excitation centred at
+2 kHz, offsets across ±10 kHz.
"""

import numpy as np

import nutate as nt

T, N, SHIFT = 200e-6, 512, 2e3


def gaussian_90(duration: float, n: int, shift: float):
    """I, Q (normalised to 1), duration (s) and Rabi frequency (Hz) of a 90° Gaussian
    exciting around ``shift`` Hz."""
    t = np.linspace(-duration / 2, duration / 2, n)
    envelope = np.exp(-((6 * t / duration) ** 2))  # cut at ±3σ·√2
    rabi = 0.25 / (envelope.mean() * duration)  # area ν1·∫envelope dt = 1/4 turn = 90°
    phase = 2 * np.pi * shift * t  # dφ/dt = shift moves the band to +shift
    return envelope * np.cos(phase), envelope * np.sin(phase), duration, rabi


def main() -> None:
    i, q, duration, rabi = gaussian_90(T, N, SHIFT)
    pulse = nt.waveform(i=i, q=q, duration=duration, rabi=rabi, name="Gaussian 90°")
    result = nt.run(
        pulse,
        20e3,
        name="waveform",
        still=True,
        b1=True,
        out="output",
        video_options={"inset_phase": True},
    )
    print(*result.files, sep="\n")


if __name__ == "__main__":
    main()
