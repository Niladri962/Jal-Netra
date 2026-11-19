"""LoRa / LoRaWAN radio link + gateway model (IN865 band).

* Link budget: log-distance rural path loss with per-node shadowing and
  per-packet fading; SNR compared with the SX127x demodulation floor per SF.
* Spreading factor chosen per node like ADR (lowest SF with >= 8 dB margin).
* Time-on-air from the Semtech SX1276 datasheet formula.
* Gateway keeps a store-and-forward buffer: if its 4G backhaul is down,
  frames are timestamped on reception and forwarded when the link returns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .scenario import EventSpec, Scenario

TX_DBM = 14.0
NODE_ANT_DBI = 2.0
CABLE_LOSS_DB = 1.0
NOISE_FLOOR_DBM = -174 + 10 * math.log10(125e3) + 6      # ~ -117 dBm (NF 6 dB)
SNR_REQ = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
LORAWAN_OVERHEAD = 13                                     # MHDR+FHDR+FPort+MIC
LINK_MARGIN_DB = 8.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def path_loss_db(d_km: float) -> float:
    """Rural log-distance model (n = 3.3), elevated gateway antenna."""
    return 120.0 + 33.0 * math.log10(max(d_km, 0.02))


def airtime_ms(payload_bytes: int, sf: int, bw: float = 125e3, cr: int = 1,
               preamble: int = 8, explicit_header: bool = True, crc: bool = True) -> float:
    t_sym = (2 ** sf) / bw
    de = 1 if (sf >= 11 and bw == 125e3) else 0
    h = 0 if explicit_header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * int(crc) - 20 * h
    n_payload = 8 + max(math.ceil(num / (4 * (sf - 2 * de))) * (cr + 4), 0)
    return ((preamble + 4.25) + n_payload) * t_sym * 1000


@dataclass
class Link:
    node_id: str
    distance_km: float
    shadow_db: float
    sf: int
    rssi_mean: float
    snr_mean: float
    airtime_ms: float
    sent: int = 0
    received: int = 0
    antenna: str = "2 dBi whip"
    nb_trans: int = 1
    rssi_log: list = field(default_factory=list)

    @property
    def pdr(self) -> float:
        return self.received / self.sent if self.sent else 0.0


@dataclass
class Uplink:
    node_id: str
    payload: bytes
    gw_time: datetime        # sample timestamp assigned by the gateway on reception
    arrival: datetime        # when the cloud receives it (later if backhaul is down)
    rssi: float
    snr: float
    sf: int


class RadioNetwork:
    def __init__(self, sc: Scenario, rng: np.random.Generator, payload_len: int):
        self.sc = sc
        self.rng = rng
        self.links: dict[str, Link] = {}
        phy_len = payload_len + LORAWAN_OVERHEAD
        gw = sc.gateway
        for n in sc.nodes:
            d = haversine_km(n.lat, n.lon, gw.lat, gw.lon)
            shadow = rng.normal(0, 4)
            antenna = "2 dBi whip"
            ant_gain = NODE_ANT_DBI
            rssi = TX_DBM + ant_gain + gw.antenna_gain_dbi - CABLE_LOSS_DB - path_loss_db(d) - shadow
            if rssi - NOISE_FLOOR_DBM - SNR_REQ[12] < LINK_MARGIN_DB:
                # network-planning step: weak site gets a directional Yagi pointed at the gateway
                antenna, ant_gain = "8 dBi Yagi (planner upgrade)", 8.0
                rssi += ant_gain - NODE_ANT_DBI
            snr = rssi - NOISE_FLOOR_DBM
            sf = next((s for s in range(7, 13) if snr - SNR_REQ[s] >= LINK_MARGIN_DB), 12)
            nb_trans = 2 if sf >= 11 else 1          # LoRaWAN NbTrans: repeat frames on long-range links
            link = Link(n.id, d, shadow, sf, rssi, min(snr, 10.0), airtime_ms(phy_len, sf) * nb_trans)
            link.antenna, link.nb_trans = antenna, nb_trans
            self.links[n.id] = link
        self.outages = [e for e in sc.events if e.type == "gateway_outage"]

    def uplink(self, node_id: str, payload: bytes, t: datetime) -> Uplink | None:
        link = self.links[node_id]
        link.sent += 1
        for _ in range(link.nb_trans):
            fade = self.rng.normal(0, 3.5)
            snr = link.snr_mean + fade
            if snr >= SNR_REQ[link.sf] and self.rng.random() >= 0.005:   # 0.5% interference / collisions
                break
        else:
            return None
        link.received += 1
        rssi = link.rssi_mean + fade
        link.rssi_log.append(rssi)
        arrival = t
        for ev in self.outages:
            if ev.start <= t < ev.end:
                arrival = ev.end           # held in gateway buffer, flushed when backhaul returns
        return Uplink(node_id, payload, t, arrival, round(rssi, 1), round(snr, 1), link.sf)

    def summary(self) -> list[dict]:
        rows = []
        for nid, l in self.links.items():
            rows.append({"node": nid, "distance_km": round(l.distance_km, 2), "sf": l.sf,
                         "antenna": l.antenna, "nb_trans": l.nb_trans,
                         "rssi_dbm": round(float(np.mean(l.rssi_log)) if l.rssi_log else l.rssi_mean, 1),
                         "snr_db": round(l.snr_mean, 1), "airtime_ms": round(l.airtime_ms, 1),
                         "sent": l.sent, "received": l.received, "pdr": round(l.pdr, 4),
                         "duty_cycle_pct": round(l.airtime_ms / (self.sc.step_min * 60 * 1000) * 100, 4)})
        return rows
