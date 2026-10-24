"""Molmo2 point markup with and without the leading image index."""

from __future__ import annotations

from conftest import ROOT  # noqa: F401

from rsched.molmo_server import parse_point


def test_parse_point_formats():
    # G2 N=1, 'the bowl': the true point is (418, 494), not (1, 418).
    assert parse_point('<points coords="1 1 418 494">bowl</points><|im_end|>') == (418.0, 494.0)
    assert parse_point('<points coords="1 1 229 541">stove burner</points>') == (229.0, 541.0)
    # RPent's own cases still hold.
    assert parse_point('<points coords="1 125 875"/>') == (125.0, 875.0)
    assert parse_point('<points coords="1 125 875; 2 500 600"/>') == (125.0, 875.0)
    assert parse_point('<points coords="1 1 125 875 2 500 600"/>') == (125.0, 875.0)
    assert parse_point("This isn't in the image.") is None
    assert parse_point('<points coords="1 1001 200"/>') is None
