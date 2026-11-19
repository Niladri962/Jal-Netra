"""Cloud backend: ingest LoRaWAN uplinks, decode, persist to SQLite."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from .codec import decode, orp_to_free_chlorine
from .lora import Uplink
from .scenario import Scenario

SCHEMA = """
CREATE TABLE IF NOT EXISTS uplinks (
    id INTEGER PRIMARY KEY, node TEXT, seq INTEGER, gw_time TEXT, arrival TEXT,
    rssi REAL, snr REAL, sf INTEGER, payload_hex TEXT);
CREATE TABLE IF NOT EXISTS readings (
    ts TEXT, node TEXT, ph REAL, orp_mv INTEGER, cl_est REAL, turb REAL, tds INTEGER,
    flow REAL, temp REAL, batt_mv INTEGER, flags INTEGER, PRIMARY KEY (ts, node));
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY, time TEXT, data_ts TEXT, type TEXT, severity TEXT, location TEXT,
    village TEXT, title TEXT, message TEXT, message_te TEXT, status TEXT, resolved_time TEXT);
CREATE TABLE IF NOT EXISTS outbox (
    time TEXT, alert_id TEXT, channel TEXT, recipient TEXT, text TEXT, segments INTEGER, status TEXT);
"""


class Store:
    def __init__(self, path: Path, sc: Scenario):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.by_num = {n.num: n.id for n in sc.nodes}
        self._n = 0

    def ingest(self, ups: list[Uplink]) -> list[dict]:
        out, up_rows, rd_rows = [], [], []
        for u in ups:
            r = decode(u.payload)
            nid = self.by_num[r["node"]]
            r["node_id"] = nid
            r["ts"] = u.gw_time
            r["arrival"] = u.arrival
            r["rssi"], r["snr"] = u.rssi, u.snr
            r["cl_est"] = orp_to_free_chlorine(r["orp_mv"], r["ph"])
            out.append(r)
            up_rows.append((nid, r["seq"], u.gw_time.isoformat(), u.arrival.isoformat(), u.rssi, u.snr, u.sf,
                            u.payload.hex()))
            rd_rows.append((u.gw_time.isoformat(), nid, r["ph"], r["orp_mv"], round(r["cl_est"], 3), r["turb"],
                            r["tds"], r["flow"], r["temp"], r["batt_mv"], r["flags"]))
        if up_rows:
            self.db.executemany("INSERT INTO uplinks (node,seq,gw_time,arrival,rssi,snr,sf,payload_hex) "
                                "VALUES (?,?,?,?,?,?,?,?)", up_rows)
            self.db.executemany("INSERT OR REPLACE INTO readings VALUES (?,?,?,?,?,?,?,?,?,?,?)", rd_rows)
            self._n += 1
            if self._n % 200 == 0:
                self.db.commit()
        return out

    def save_alerts(self, alerts, outbox) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(a.id, a.time.isoformat(), a.data_ts.isoformat(), a.type, a.severity, a.location, a.village,
              a.title, a.message, a.message_te, a.status,
              a.resolved_time.isoformat() if a.resolved_time else None) for a in alerts])
        self.db.executemany("INSERT INTO outbox VALUES (?,?,?,?,?,?,?)",
                            [(o["time"], o["alert_id"], o["channel"], o["recipient"], o["text"], o["segments"],
                              o["status"]) for o in outbox])
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M")
