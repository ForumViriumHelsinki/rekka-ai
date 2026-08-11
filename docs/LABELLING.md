# Labelling — the short version

The card to keep open while reviewing. The reasoning behind these rules is in
`docs/DESIGN.md` §5, which wins if the two ever disagree; guidance for one
particular area is in its `notes` in `aois/helsinki.yaml`, shown beside the map.

## The four rules that decide the metrics

1. **Label every real vehicle.** An unlabelled vehicle is not "skipped", it is
   *background* — and the model's correct detection of it scores as a false
   positive. Reject means "not a vehicle", never "I can't tell what kind".
2. **One vehicle, one box.** Two boxes on one object (a cab and its body, a
   nose and its trailer) teach the model to fire twice and inflate every count.
   Reject the extra one.
3. **Call the same thing the same way everywhere.** A convention that changes
   between areas — or between train and validation — makes the eval measure
   your inconsistency instead of the model.
4. **Only `truck` is gated.** Ship gates are truck recall ≥ 0.90 and precision
   ≥ 0.85. `bus`, `van` and `car` are reported and never gate. So: think hard
   before calling something a truck; decide van-vs-car in two seconds.

## Class: read the measurement, not the pixels

The footer prints length × width for the selected box. That number is reliable
even when the imagery is a smear. Measured over the 2,866 labels so far:

| class | length p25–p75 | median | width p25–p75 | n |
|---|---|---|---|---|
| `car` | 4.6–5.0 m | 4.83 m | 2.2–2.4 m | 1961 |
| `van` | 5.2–6.1 m | 5.60 m | 2.3–2.5 m | 381 |
| `truck` | 8.0–11.3 m | 9.84 m | 2.9–3.2 m | 370 |
| `bus` | 12.8–14.8 m | 14.52 m | 2.9–3.1 m | 154 |

**Truck or van is decided by shape, not length.** A truck — including the
small ones — has a **cab that stops and a load body that starts**: boxy at the
rear, open or closed. A van is one continuous shell from windscreen to rear
doors. Look for the break.

Length only helps at the ends: under 6.5 m a separated body is rare, at 8 m
and over it is a truck 293 times out of 294. **Between 6.5 and 8.0 m the
length tells you nothing** — read the rear. In the 5–6 m car/van overlap this
project calls it `car` three times in four.

## Keep or drop

| box it | leave it |
|---|---|
| Box trucks, straight trucks (*kuorma-autot*) | Full trailers (*perävaunut*) — coupled or not |
| Semi-trailer rigs — **tractor + trailer in one box** | Bare platforms / swap bodies (*lavat*) |
| Concrete mixers, platforms **with** a cab | Shipping containers |
| Buses and coaches | Boats, and hulls on cradles |
| Vans, and cars (a pickup is a `car`) | Rail rolling stock |
| | Motorhomes, RVs, caravans |

## The confusers that actually turn up

- **Truck + full trailer** (the common Finnish rig) — box the truck unit; the
  trailer is a negative, coupled or not. A semi-trailer rig is the opposite
  case and stays one box: it cannot stand without its tractor.
- **Motorhomes, RVs and caravans** — reject. A white 6–8 m box body reads as a
  van at a glance and they are not only in campsites: the round-1 audit found
  them parked in ordinary industrial car parks in four areas. If it is white,
  6–8 m, and has no cab you can point at, look twice before calling it a van.
- **Pickups** — `car`. They are used like cars and measure like cars.
- **Boat on a cradle** — the most truck-like shape in the collection. Reject.
- **Rail stock beside a yard** — a carriage is a long oriented rectangle.
  Reject.
- **Bobtail tractor unit** — under 6 m but still a truck; the length gate
  alone does not settle it.

## When you genuinely cannot tell

- **Clearly a vehicle, class unreadable** → pick the likeliest class. Never
  reject it: that teaches background *and* costs precision.
- **Cannot tell it is a vehicle at all** → reject only if you would bet against
  it. A wrong car is cheap; a rejected real vehicle is not.
- **Reject** what the detector proposed; **Del** only boxes you drew yourself.
  A reject keeps the file a complete record of what was proposed.

## Geometry: do not over-invest

Draw the centreline — nose, then tail, then scroll for width. Length and
heading are recomputed from the box, so ±0.2 m of box edge changes nothing:
the gates measure counts per site, not box perfection. Spend the saved time on
finding the vehicles the detector missed, which is where recall comes from.

## Keys

| key | |
|---|---|
| `T` `B` `V` `C` | truck / bus / van / car |
| `X` | reject |
| `N` `P` | next / previous unreviewed |
| `D` | draw: nose, tail, scroll width, click or `Enter` |
| `G` | go to a box by its number in the file |
| `↑` `↓` `←` `→` | length / rotation of the selected box (`⇧` coarse) |
| `⇧` scroll | width of the selected box |
| `Del` | delete a box you drew |
| `⌘/⌃ Z` | undo |

## Before calling an area done

```sh
uv run rekka-ai progress     # N/N reviewed, and any schema problem
```

Every candidate needs a verdict — `export` refuses to run while one is
unreviewed, and refuses again if a box's centre has drifted outside its area.
