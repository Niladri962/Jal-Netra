"""17-byte uplink payload - identical layout to firmware/jalnetra_node/jalnetra_node.ino.

    byte  field        type   scale            range
    0     node_id      u8     -                1..255
    1-2   seq          u16    -                wraps at 65535
    3     flags        u8     bit0 low-batt, bit1 sensor self-test fail, bit2 cold boot
    4-5   ph           u16    x100             0.00 .. 14.00
    6-7   orp          i16    mV               -2000 .. +2000
    8-9   turbidity    u16    x10  NTU         0 .. 6553.5
    10-11 tds          u16    mg/L             0 .. 65535
    12-13 flow         u16    x10  L/min       0 .. 6553.5
    14-15 water temp   i16    x10  degC
    16    battery      u8     (mV-2500)/10     2500 .. 5050 mV

All multi-byte fields are big-endian (network order). No timestamp is sent:
the LoRaWAN gateway stamps every frame on reception (GPS/NTP time).
Free chlorine is NOT measured directly - it is derived in the cloud from
ORP and pH (see ``orp_to_free_chlorine``), which keeps the node cheap.
"""
from __future__ import annotations

import math
import struct

FMT = ">BHBHhHHHhB"
SIZE = struct.calcsize(FMT)
assert SIZE == 17

FLAG_LOW_BATT, FLAG_SELFTEST, FLAG_BOOT = 0x01, 0x02, 0x04


def encode(r: dict) -> bytes:
    return struct.pack(
        FMT, r["node"], r["seq"] & 0xFFFF, r["flags"], r["ph_x100"], r["orp_mv"],
        r["turb_x10"], r["tds"], r["flow_x10"], r["temp_x10"], r["batt"])


def decode(b: bytes) -> dict:
    node, seq, flags, ph, orp, turb, tds, flow, temp, batt = struct.unpack(FMT, b)
    return {"node": node, "seq": seq, "flags": flags, "ph_x100": ph, "orp_mv": orp, "turb_x10": turb,
            "tds": tds, "flow_x10": flow, "temp_x10": temp, "batt": batt,
            "ph": ph / 100, "turb": turb / 10, "flow": flow / 10, "temp": temp / 10,
            "batt_mv": 2500 + 10 * batt}


# Empirical ORP <-> free chlorine relation (shape follows published ORP/FAC curves:
# ~+140 mV per decade of chlorine, ~-40 mV per pH unit). Calibrated per probe on site
# against a DPD colorimeter; the same constants are used to synthesise ORP in the node model.
ORP_A, ORP_B, ORP_PH = 600.0, 140.0, 40.0


def free_chlorine_to_orp(cl: float, ph: float) -> float:
    return ORP_A + ORP_B * math.log10(max(cl, 0.01)) - ORP_PH * (ph - 7.0)


def orp_to_free_chlorine(orp_mv: float, ph: float) -> float:
    return 10 ** ((orp_mv - ORP_A + ORP_PH * (ph - 7.0)) / ORP_B)
