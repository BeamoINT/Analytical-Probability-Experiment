#!/usr/bin/env python3
"""Quick test of position sizing."""

from polyb0t.models.position_sizing import PositionSizer

# Test with user's balance
available_usdc = 247.66
reserved_usdc = 0.0
edge_net = 0.054  # 5.4% edge from the intents
confidence = 0.75

sizer = PositionSizer()

result = sizer.compute_size(
    edge_net=edge_net,
    confidence=confidence,
    available_usdc=available_usdc,
    reserved_usdc=reserved_usdc,
)

print(f"\n=== Position Sizing Test ===")
print(f"Available USDC: ${available_usdc:.2f}")
print(f"Reserved USDC: ${reserved_usdc:.2f}")
print(f"Edge (net): {edge_net*100:.1f}%")
print(f"Confidence: {confidence:.2f}")
print(f"\nMax % per trade: {sizer.max_pct_per_trade*100:.0f}%")
print(f"Max $ per trade: ${available_usdc * sizer.max_pct_per_trade:.2f}")
print(f"\nFinal Size: ${result.size_usd_final:.2f}")
print(f"Sizing Reason: {result.sizing_reason}")
print(f"Kelly Fraction: {result.kelly_fraction:.4f}")
print(f"\nExpected: ~$111 (45% of $247.66)")
print(f"Actual: ${result.size_usd_final:.2f}")

if result.size_usd_final > 120:
    print(f"\n❌ ERROR: Size too large! Should be ~$111 max")
else:
    print(f"\n✅ SUCCESS: Size within expected range")

