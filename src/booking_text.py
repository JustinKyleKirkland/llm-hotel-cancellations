"""Booking -> a short, deterministic description in a hotel manager's words.

Run as a script to write outputs/example_prompts.md (a few descriptions plus the exact
prompt AnyJev builds around one of them).
"""
from __future__ import annotations

import pandas as pd

from common import setup

MEALS = {"BB": "bed & breakfast", "HB": "half board (breakfast and dinner)",
         "FB": "full board", "SC": "room only (no meal package)"}
SEGMENTS = {"Online TA": "online travel agency", "Offline TA/TO": "offline travel agent / tour operator",
            "Direct": "direct", "Corporate": "corporate", "Groups": "groups",
            "Complementary": "complimentary", "Aviation": "aviation (airline crews)", "Undefined": "undefined"}
CHANNELS = {"TA/TO": "travel agents / tour operators", "Direct": "direct", "Corporate": "corporate",
            "GDS": "a global distribution system", "Undefined": "undefined"}
DEPOSITS = {"No Deposit": "no deposit was taken", "Non Refund": "a non-refundable deposit covering the full stay was paid",
            "Refundable": "a refundable deposit (less than the full stay) was paid"}
CUSTOMERS = {"Transient": "transient (an individual booking)", "Transient-Party": "transient-party (linked to another individual booking)",
             "Contract": "contract (allotment / contracted rate)", "Group": "part of a group"}


def _n(k: int, word: str, plural: str | None = None) -> str:
    return f"{k} {word if k == 1 else (plural or word + 's')}"


def describe(b) -> str:
    """b: a row (Series or dict) of the cleaned booking table."""
    hotel = "a city hotel in Lisbon" if b["hotel"] == "City Hotel" else "a resort hotel in the Algarve"
    lead = int(b["lead_time"])
    lead_txt = "on the day of arrival" if lead == 0 else f"{_n(lead, 'day')} before arrival"
    we, wk = int(b["stays_in_weekend_nights"]), int(b["stays_in_week_nights"])
    nights = we + wk
    stay = f"{_n(nights, 'night')} ({_n(we, 'weekend night')}, {_n(wk, 'weeknight')})" if nights else "a day use (0 nights)"
    adults, kids, babies = int(b["adults"]), int(b["children"]), int(b["babies"])
    party = [_n(adults, "adult")]
    if kids:
        party.append(_n(kids, "child", "children"))
    if babies:
        party.append(_n(babies, "baby", "babies"))
    party_txt = " and ".join([", ".join(party[:-1]), party[-1]] if len(party) > 1 else party)
    if not (kids or babies):
        party_txt += ", no children"
    repeat = "a returning guest" if int(b["is_repeated_guest"]) else "a first-time guest"
    prev_c = int(b["previous_cancellations"])
    prev_ok = int(b["previous_bookings_not_canceled"])
    history = (f"This guest has {_n(prev_c, 'previous cancellation')} and "
               f"{_n(prev_ok, 'previous booking')} that were not canceled.")
    changes = int(b["booking_changes"])
    changes_txt = "The booking has not been changed since it was made." if changes == 0 else \
        f"The booking has been changed {_n(changes, 'time')} since it was made."
    parking = int(b["required_car_parking_spaces"])
    parking_txt = "no parking" if parking == 0 else _n(parking, "parking space")
    req = int(b["total_of_special_requests"])
    req_txt = "no special requests" if req == 0 else _n(req, "special request")
    return (
        f"Booking at {hotel}, made {lead_txt}. "
        f"Arrival in {b['arrival_date_month']} for {stay}. "
        f"Party: {party_txt}. Meal plan: {MEALS.get(b['meal'], b['meal'])}. "
        f"Market segment: {SEGMENTS.get(b['market_segment'], b['market_segment'])}; "
        f"distribution channel: {CHANNELS.get(b['distribution_channel'], b['distribution_channel'])}. "
        f"The customer is {repeat}. {history} {changes_txt} "
        f"Deposit: {DEPOSITS.get(b['deposit_type'], b['deposit_type'])}. "
        f"Customer type: {CUSTOMERS.get(b['customer_type'], b['customer_type'])}. "
        f"Average daily rate: EUR {float(b['adr']):.2f}. "
        f"The guest asked for {parking_txt} and made {req_txt}."
    )


def main():
    cfg = setup("Write example booking descriptions")
    from anyjev import Question
    from anyjev.readout import DEFAULT_SYSTEM, build_prompt

    test = pd.read_csv(cfg["data"]["processed_dir"] / "test.csv")
    # a few varied examples: first canceled/non-canceled from each hotel, plus a non-refundable one
    picks = []
    for hotel in ("City Hotel", "Resort Hotel"):
        for y in (1, 0):
            picks.append(test[(test.hotel == hotel) & (test.is_canceled == y)].iloc[0])
    picks.append(test[test.deposit_type == "Non Refund"].iloc[0])
    q = Question.noul(cfg["question"], name="cancel")
    spec = build_prompt(describe(picks[0]), q, [0, 1], DEFAULT_SYSTEM)
    spec_rev = build_prompt(describe(picks[0]), q, [1, 0], DEFAULT_SYSTEM)
    lines = ["# Example prompts", "",
             "Each booking is turned into text by `src/booking_text.py::describe` (deterministic: the same row "
             "always gives the same text). AnyJev then wraps it in its own prompt.", ""]
    for i, b in enumerate(picks, 1):
        lines += [f"## Example {i}: {b['hotel']}, booking_id {b['booking_id']} "
                  f"(actual outcome: {'canceled' if b['is_canceled'] else 'not canceled'})", "",
                  "> " + describe(b), ""]
    lines += ["## The full prompt AnyJev sends for Example 1", "",
              "System message:", "", "```", spec.system, "```", "", "User message (phrasing 1, used by the naive readout):",
              "", "```", spec.user, "```", "",
              "L0 also reads the second phrasing (`Answer No or Yes.`) and combines the two:", "",
              "```", spec_rev.user.splitlines()[-1], "```", "",
              "The model's chat template is applied on top (thinking disabled), and AnyJev reads the "
              "probabilities of the `Yes` / `No` tokens at the next position. Nothing is generated.", ""]
    path = cfg["paths"]["outputs"] / "example_prompts.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")
    print(describe(picks[0]))


if __name__ == "__main__":
    main()
