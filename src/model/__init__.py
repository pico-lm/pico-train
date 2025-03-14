"""
Model Package

This Package contains Pico models (currently only the Pico Decoder). We plan to implement other
architectures in the future.

If you have other models you'd like to implement, we recommend you add modules to this package.
"""

from .pico_decoder import PicoDecoder

__all__ = ["PicoDecoder"]
