# Example prompts

Each booking is turned into text by `src/booking_text.py::describe` (deterministic: the same row always gives the same text). AnyJev then wraps it in its own prompt.

## Example 1: City Hotel, booking_id 50260 (actual outcome: canceled)

> Booking at a city hotel in Lisbon, made 295 days before arrival. Arrival in April for 2 nights (0 weekend nights, 2 weeknights). Party: 2 adults, no children. Meal plan: bed & breakfast. Market segment: groups; distribution channel: travel agents / tour operators. The customer is a first-time guest. This guest has 0 previous cancellations and 0 previous bookings that were not canceled. The booking has not been changed since it was made. Deposit: a non-refundable deposit covering the full stay was paid. Customer type: transient (an individual booking). Average daily rate: EUR 62.00. The guest asked for no parking and made no special requests.

## Example 2: City Hotel, booking_id 94659 (actual outcome: not canceled)

> Booking at a city hotel in Lisbon, made 13 days before arrival. Arrival in August for 3 nights (2 weekend nights, 1 weeknight). Party: 2 adults, no children. Meal plan: half board (breakfast and dinner). Market segment: direct; distribution channel: direct. The customer is a first-time guest. This guest has 0 previous cancellations and 0 previous bookings that were not canceled. The booking has been changed 1 time since it was made. Deposit: no deposit was taken. Customer type: transient (an individual booking). Average daily rate: EUR 204.00. The guest asked for 1 parking space and made no special requests.

## Example 3: Resort Hotel, booking_id 8862 (actual outcome: canceled)

> Booking at a resort hotel in the Algarve, made 30 days before arrival. Arrival in October for 4 nights (1 weekend night, 3 weeknights). Party: 1 adult, no children. Meal plan: bed & breakfast. Market segment: online travel agency; distribution channel: travel agents / tour operators. The customer is a first-time guest. This guest has 0 previous cancellations and 0 previous bookings that were not canceled. The booking has not been changed since it was made. Deposit: no deposit was taken. Customer type: transient (an individual booking). Average daily rate: EUR 57.25. The guest asked for no parking and made no special requests.

## Example 4: Resort Hotel, booking_id 22296 (actual outcome: not canceled)

> Booking at a resort hotel in the Algarve, made 1 day before arrival. Arrival in December for 1 night (0 weekend nights, 1 weeknight). Party: 1 adult, no children. Meal plan: bed & breakfast. Market segment: corporate; distribution channel: corporate. The customer is a returning guest. This guest has 0 previous cancellations and 5 previous bookings that were not canceled. The booking has not been changed since it was made. Deposit: no deposit was taken. Customer type: transient (an individual booking). Average daily rate: EUR 63.00. The guest asked for no parking and made no special requests.

## Example 5: City Hotel, booking_id 50260 (actual outcome: canceled)

> Booking at a city hotel in Lisbon, made 295 days before arrival. Arrival in April for 2 nights (0 weekend nights, 2 weeknights). Party: 2 adults, no children. Meal plan: bed & breakfast. Market segment: groups; distribution channel: travel agents / tour operators. The customer is a first-time guest. This guest has 0 previous cancellations and 0 previous bookings that were not canceled. The booking has not been changed since it was made. Deposit: a non-refundable deposit covering the full stay was paid. Customer type: transient (an individual booking). Average daily rate: EUR 62.00. The guest asked for no parking and made no special requests.

## The full prompt AnyJev sends for Example 1

System message:

```
You are a decision function. You will be given a state and one question. Reply with the answer label only: no words, no punctuation, no explanation.
```

User message (phrasing 1, used by the naive readout):

```
State:
Booking at a city hotel in Lisbon, made 295 days before arrival. Arrival in April for 2 nights (0 weekend nights, 2 weeknights). Party: 2 adults, no children. Meal plan: bed & breakfast. Market segment: groups; distribution channel: travel agents / tour operators. The customer is a first-time guest. This guest has 0 previous cancellations and 0 previous bookings that were not canceled. The booking has not been changed since it was made. Deposit: a non-refundable deposit covering the full stay was paid. Customer type: transient (an individual booking). Average daily rate: EUR 62.00. The guest asked for no parking and made no special requests.

Question: Based on this booking, will the guest cancel it before arrival?
Answer Yes or No.
```

L0 also reads the second phrasing (`Answer No or Yes.`) and combines the two:

```
Answer No or Yes.
```

The model's chat template is applied on top (thinking disabled), and AnyJev reads the probabilities of the `Yes` / `No` tokens at the next position. Nothing is generated.
