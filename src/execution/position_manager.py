"""
Position manager — the trailing-stop exit logic.

Once a position is open, this decides on each completed bar whether to exit and
where the stop now sits. It is the "let winners run, cut losses fast" mechanism:
the stop trails upward behind the price's high-water mark and never moves down.

Pure logic: no money, no network, no executor dependency beyond the Position type.
The trading loop will call process_bar each time a new bar closes for an open
position, then act on the returned ExitDecision via the executor.

Trail distance is FROZEN at entry (= entry - initial_stop = 1.5*ATR at entry),
stored on the Position. We do not recompute ATR per bar.

Within-bar ordering is deliberately PESSIMISTIC: we check the stop against the
bar's LOW using the CURRENT stop first; only if the stop survived do we then
raise the trail using the bar's HIGH. This prevents the look-ahead bug where one
bar both ratchets the stop up and dips down (which should have exited first).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from src.execution.executor import Position


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExitDecision:
    """Result of processing one bar against an open position."""

    should_exit: bool
    exit_price: Decimal | None = None   # the stop level, if exiting
    reason: str = ""


def process_bar(
    position: Position,
    bar_high: Decimal,
    bar_low: Decimal,
) -> ExitDecision:
    """
    Process one completed bar against an open position.

    Order of operations (pessimistic, no look-ahead):
      1. If bar_low <= current_stop, the stop was hit during the bar -> EXIT at the stop.
      2. Otherwise, raise the trail: high_water_mark = max(hwm, bar_high), and
         current_stop = max(current_stop, hwm - trail_distance). The stop only
         ever ratchets UP.

    Mutates the position's high_water_mark and current_stop in place (when not exiting).
    """
    # Step 1: did the stop get hit this bar? Check against the stop coming INTO the bar.
    if bar_low <= position.current_stop:
        logger.info(
            f"EXIT {position.product_id}: bar low ${bar_low:.4f} <= stop "
            f"${position.current_stop:.4f}"
        )
        return ExitDecision(
            should_exit=True,
            exit_price=position.current_stop,
            reason="trailing_stop_hit",
        )

    # Step 2: stop survived — raise the trail using this bar's high.
    if bar_high > position.high_water_mark:
        position.high_water_mark = bar_high
        new_stop = position.high_water_mark - position.trail_distance
        if new_stop > position.current_stop:
            old_stop = position.current_stop
            position.current_stop = new_stop
            logger.debug(
                f"TRAIL {position.product_id}: new high ${bar_high:.4f}, "
                f"stop raised ${old_stop:.4f} -> ${new_stop:.4f}"
            )

    return ExitDecision(should_exit=False)


# ============================================================
# Self-test
# ============================================================


if __name__ == "__main__":
    """
    Verify the trailing stop: initial stop protects, trail ratchets up, exits on
    reversal, and the stop never moves down.
        python3 -m src.execution.position_manager
    """
    import sys
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    failed = False

    def expect(name: str, condition: bool) -> None:
        global failed
        print(f"  {'✓' if condition else '✗'} {name}")
        if not condition:
            failed = True

    def fresh_position() -> Position:
        # Entry 100, initial stop 94 -> trail distance 6 (i.e. 1.5*ATR with ATR=4)
        return Position(
            product_id="BTC-USD",
            entry_price=Decimal("100"),
            quantity=Decimal("0.3"),
            entry_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            entry_cost=Decimal("30"),
            initial_stop=Decimal("94"),
            trail_distance=Decimal("6"),
            current_stop=Decimal("94"),
            high_water_mark=Decimal("100"),
        )

    # --- Test 1: initial stop protects the downside ---
    print("Test 1: Price drops to stop immediately -> exit at 94")
    pos = fresh_position()
    d = process_bar(pos, bar_high=Decimal("100"), bar_low=Decimal("93"))
    expect("should_exit", d.should_exit)
    expect("exit price == 94", d.exit_price == Decimal("94"))
    print()

    # --- Test 2: price rises, trail ratchets up, no exit ---
    print("Test 2: Bar high 110 (low 99) -> no exit, stop trails to 104")
    pos = fresh_position()
    d = process_bar(pos, bar_high=Decimal("110"), bar_low=Decimal("99"))
    expect("no exit", not d.should_exit)
    expect("high_water_mark == 110", pos.high_water_mark == Decimal("110"))
    expect("stop raised to 104 (110 - 6)", pos.current_stop == Decimal("104"))
    print()

    # --- Test 3: after trailing up, a reversal exits at the raised stop (in profit) ---
    print("Test 3: Next bar low 103 (<= raised stop 104) -> exit at 104, in profit")
    # pos still has current_stop 104 from Test 2
    d = process_bar(pos, bar_high=Decimal("106"), bar_low=Decimal("103"))
    expect("should_exit", d.should_exit)
    expect("exit price == 104 (above entry 100 = profit)", d.exit_price == Decimal("104"))
    print()

    # --- Test 4: stop never moves DOWN ---
    print("Test 4: A lower-high bar does NOT lower the stop")
    pos = fresh_position()
    process_bar(pos, bar_high=Decimal("110"), bar_low=Decimal("99"))   # stop -> 104
    stop_before = pos.current_stop
    process_bar(pos, bar_high=Decimal("108"), bar_low=Decimal("105"))  # lower high, no stop hit
    expect("stop unchanged at 104", pos.current_stop == stop_before == Decimal("104"))
    print()

    # --- Test 5: pessimistic ordering — a bar that both spikes up AND dips to stop exits ---
    print("Test 5: Bar dips to stop AND spikes high -> exits (stop checked first)")
    pos = fresh_position()  # stop 94
    d = process_bar(pos, bar_high=Decimal("120"), bar_low=Decimal("94"))
    expect("exits at 94 (not saved by the high spike)", d.should_exit and d.exit_price == Decimal("94"))
    print()

    if failed:
        print("✗ Some tests failed.")
        sys.exit(1)
    print("All tests passed. ✓")