"""Alert manager: de-duplication, cool-downs, routing, bilingual SMS / WhatsApp dispatch.

Messages go out in English + Telugu. SMS containing Telugu is UCS-2 encoded
(67 chars per concatenated segment), so the segment count is tracked for cost.
The Telugu templates are a first draft and should be checked by a native speaker
and the district team before field use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .scenario import Scenario

SEV_RANK = {"info": 0, "maintenance": 1, "warning": 2, "critical": 3}

# type -> (severity, title, English template, Telugu template, roles)
TEMPLATES: dict[str, tuple] = {
    "CONTAMINATION_RISK": (
        "critical", "Contamination risk",
        "{loc}: water-borne disease risk {mri:.0f}/100. Free chlorine {cl:.2f} mg/L, turbidity {turb:.1f} NTU. "
        "Issue boil-water advisory, flush & super-chlorinate line, ASHA to stock ORS.",
        "{loc}: నీరు కలుషితమయ్యే ప్రమాదం ({mri:.0f}/100). క్లోరిన్ {cl:.2f} mg/L, టర్బిడిటీ {turb:.1f} NTU. "
        "నీటిని కాచి చల్లార్చి తాగండి.",
        ["gp_secretary", "asha", "phc_mo", "mb_ae"]),
    "SOURCE_CHLORINATION_FAILURE": (
        "critical", "Chlorination failure at source",
        "{loc}: free chlorine {cl:.2f} mg/L at WTP outlet (norm >= 1.0). Check dosing pump. "
        "All downstream villages will receive unchlorinated water in {lag}.",
        "{loc}: శుద్ధి కేంద్రంలో క్లోరినేషన్ విఫలమైంది (క్లోరిన్ {cl:.2f} mg/L). దిగువ గ్రామాలకు హెచ్చరిక.",
        ["wtp_operator", "mb_dee", "dmho", "gp_secretary*"]),
    "LOW_CHLORINE": (
        "warning", "Low residual chlorine",
        "{loc}: free chlorine {cl:.2f} mg/L (< 0.2 mg/L BIS minimum at tap). Increase dosing / flush line.",
        "{loc}: నీటిలో క్లోరిన్ తక్కువగా ఉంది ({cl:.2f} mg/L).",
        ["mb_ae", "gp_secretary"]),
    "WATER_LOSS": (
        "warning", "Water loss detected",
        "{loc}: water-balance gap {excess:.0f} L/min ({pct:.0f}% of inflow, ~{kld:.0f} kL/day). "
        "Inspect segment between {parent} and its outlets.",
        "{loc}: పైప్‌లైన్‌లో నీటి నష్టం - నిమిషానికి సుమారు {excess:.0f} లీటర్లు. తనిఖీ చేయండి.",
        ["mb_ae", "gp_secretary"]),
    "LEAK_CONFIRMED": (
        "critical", "Leak confirmed (minimum night flow)",
        "{loc}: loss persists through 00:00-05:00 low-demand hours -> physical leak. "
        "Night excess {excess:.0f} L/min, ~{kld:.0f} kL/day lost. Dispatch repair crew.",
        "{loc}: రాత్రి వేళల్లోనూ నీరు పోతోంది - పైప్‌లైన్ లీకేజీ నిర్ధారణ. మరమ్మతు బృందాన్ని పంపండి.",
        ["mb_ae", "gp_secretary", "mb_dee"]),
    "THEFT_SUSPECTED": (
        "warning", "Unauthorised draw suspected",
        "{loc}: recurring loss of ~{excess:.0f} L/min starting ~{hour} on {count} evenings; night flow normal "
        "-> pattern of an illegal connection / motor. Field inspection at that hour advised.",
        "{loc}: ప్రతి రోజు {hour} సమయంలో నీటి నష్టం - అక్రమ కనెక్షన్ అనుమానం. తనిఖీ చేయండి.",
        ["mb_dee", "mb_ae", "mpdo"]),
    "HIGH_TDS": (
        "warning", "High dissolved solids",
        "{loc}: TDS {tds:.0f} mg/L (BIS acceptable 500). Possible groundwater / borewell mixing at reservoir.",
        "{loc}: నీటిలో లవణాలు (TDS) ఎక్కువగా ఉన్నాయి ({tds:.0f} mg/L).",
        ["mb_ae", "gp_secretary"]),
    "PH_OUT_OF_RANGE": (
        "warning", "pH out of range",
        "{loc}: pH {ph:.2f} outside 6.5-8.5.",
        "{loc}: నీటి pH {ph:.2f} - పరిధి దాటింది.",
        ["mb_ae"]),
    "EQUITY_SHORTFALL": (
        "warning", "Supply equity shortfall",
        "{loc}: received {lpcd:.0f} LPCD yesterday vs {target:.0f} target ({pct:.0f}%). Village median {median:.0f} LPCD. "
        "Check valve settings / tail-end pressure.",
        "{loc}: ఒక్కొక్కరికి రోజుకు {lpcd:.0f} లీటర్లు మాత్రమే అందుతున్నాయి (లక్ష్యం {target:.0f}).",
        ["mb_ae", "mpdo", "gp_secretary"]),
    "SENSOR_FAULT": (
        "maintenance", "Sensor fault",
        "{loc}: {sensor} reading frozen for {n} samples - excluded from risk scoring. Clean / recalibrate probe.",
        "{loc}: {sensor} సెన్సార్ పనిచేయడం లేదు - నిర్వహణ అవసరం.",
        ["technician"]),
    "NODE_OFFLINE": (
        "maintenance", "Node offline",
        "{loc}: no data for {mins} min while gateway is healthy. Check power / solar / antenna.",
        "{loc}: నోడ్ నుండి డేటా రావడం లేదు.",
        ["technician", "gp_secretary"]),
    "GATEWAY_OFFLINE": (
        "maintenance", "Gateway backhaul down",
        "{loc}: no uplinks for {mins} min from any node. Gateway is buffering locally; check 4G / power.",
        "{loc}: గేట్‌వే ఆఫ్‌లైన్‌లో ఉంది.",
        ["technician", "mb_ae"]),
}


class _Safe(dict):
    def __missing__(self, k):
        return "?"


@dataclass
class Alert:
    id: str
    time: datetime
    data_ts: datetime
    type: str
    severity: str
    location: str
    location_name: str
    village: str
    title: str
    message: str
    message_te: str
    details: dict
    recipients: list = field(default_factory=list)
    status: str = "open"
    resolved_time: datetime | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "time": self.time.strftime("%Y-%m-%d %H:%M"),
                "data_ts": self.data_ts.strftime("%Y-%m-%d %H:%M"), "type": self.type,
                "severity": self.severity, "location": self.location, "location_name": self.location_name,
                "village": self.village, "title": self.title, "message": self.message,
                "message_te": self.message_te, "recipients": self.recipients, "status": self.status,
                "resolved_time": self.resolved_time.strftime("%Y-%m-%d %H:%M") if self.resolved_time else None,
                "details": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.details.items()
                            if isinstance(v, (int, float, str))}}


class AlertManager:
    COOLDOWN = timedelta(hours=6)

    def __init__(self, sc: Scenario, rng: np.random.Generator):
        self.sc = sc
        self.rng = rng
        self.alerts: list[Alert] = []
        self.outbox: list[dict] = []
        self.open: dict[tuple, Alert] = {}
        self.last: dict[tuple, Alert] = {}
        self.suppressed = 0

    # ------------------------------------------------------------ recipients
    def _recipients(self, roles: list[str], village: str) -> list[str]:
        out = []
        villages = sorted({n.village for n in self.sc.nodes if n.kind != "source"})
        for role in roles:
            fan_out = role.endswith("*")
            role = role.rstrip("*")
            role = self.sc.role_alias.get(role, role)
            keys = [f"{role}:{v}" for v in villages] if fan_out else [f"{role}:{village}", role]
            for k in keys:
                if k in self.sc.contacts and self.sc.contacts[k] not in out:
                    out.append(self.sc.contacts[k])
                    if not fan_out:
                        break
        return out

    # ----------------------------------------------------------------- raise
    def raise_alert(self, key: tuple, atype: str, now: datetime, data_ts: datetime, location: str,
                    details: dict, severity: str | None = None) -> Alert | None:
        if key in self.open:
            return None
        sev, title, en, te, roles = TEMPLATES[atype]
        sev = severity or sev
        prev = self.last.get((atype, location))
        if prev and now - prev.time < self.COOLDOWN and SEV_RANK[sev] <= SEV_RANK[prev.severity]:
            self.suppressed += 1
            return None
        loc_name = self.sc.location_name(location)
        village = self.sc.village_of(location)
        fmt = _Safe(details, loc=loc_name)
        a = Alert(id=f"A{len(self.alerts) + 1:04d}", time=now, data_ts=data_ts, type=atype, severity=sev,
                  location=location, location_name=loc_name, village=village, title=title,
                  message=en.format_map(fmt), message_te=te.format_map(fmt), details=details,
                  recipients=self._recipients(roles, village))
        self.alerts.append(a)
        self.open[key] = a
        self.last[(atype, location)] = a
        self._dispatch(a, now)
        return a

    def resolve(self, key: tuple, now: datetime) -> None:
        a = self.open.pop(key, None)
        if a is None:
            return
        a.status, a.resolved_time = "resolved", now
        if a.severity == "critical":
            note = f"[JalNetra] RESOLVED {a.title} - {a.location_name} back to normal at {now:%d %b %H:%M}."
            for r in a.recipients:
                self._send(now, a.id, "WhatsApp", r, note)

    def is_open(self, key: tuple) -> bool:
        return key in self.open

    def recently_open(self, atype: str, location: str, now: datetime, within: timedelta) -> bool:
        a = self.last.get((atype, location))
        if a is None:
            return False
        if a.status == "open":
            return True
        return a.resolved_time is not None and now - a.resolved_time <= within

    # --------------------------------------------------------------- dispatch
    def _send(self, now: datetime, alert_id: str, channel: str, recipient: str, text: str) -> None:
        unicode = any(ord(c) > 127 for c in text)
        per_seg = 67 if unicode else 153
        segs = 1 if len(text) <= (70 if unicode else 160) else math.ceil(len(text) / per_seg)
        status = "delivered" if self.rng.random() < 0.97 else "delivered (retry 1)"
        self.outbox.append({"time": now.strftime("%Y-%m-%d %H:%M"), "alert_id": alert_id, "channel": channel,
                            "recipient": recipient, "text": text, "segments": segs if channel == "SMS" else 0,
                            "status": status})

    def _dispatch(self, a: Alert, now: datetime) -> None:
        icon = {"critical": "RED", "warning": "AMBER", "maintenance": "MAINT", "info": "INFO"}[a.severity]
        wa = f"[JalNetra {icon}] {a.title}\n{a.message}\n{a.message_te}\nRef {a.id} | {now:%d %b %H:%M}"
        sms = f"JalNetra {icon}: {a.message_te} Ref {a.id}"
        for r in a.recipients:
            self._send(now, a.id, "WhatsApp", r, wa)
            if a.severity in ("critical", "warning"):
                self._send(now, a.id, "SMS", r, sms)
