"""Scenario definitions: network topology, people served, and injected events.

Two scenarios ship with the simulator:

* ``mandal``  - a Mission Bhagiratha segment: 1 WTP outlet, 3 village OHSRs
                (overhead service reservoirs) and 6 end-point taps, 14 days.
* ``pilot``   - the 3-node school prototype test-bed (sump outlet + 2 taps),
                4 school days, with the bench tests a student team would run.

Village names and coordinates are ILLUSTRATIVE (placed in Siddipet district,
Telangana); they do not describe the real condition of any real village.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class NodeSpec:
    id: str
    num: int                      # 1-byte on-air node id
    name: str
    village: str
    kind: str                     # "source" | "reservoir" | "tap"
    lat: float
    lon: float
    parent: str | None = None
    population: int = 0           # people (or person-equivalents) served by this branch
    lpcd_target: float = 100.0    # litres per capita per day target (Mission Bhagiratha: 100 rural)
    profile: str = "residential"  # residential | school
    supply_factor: float = 1.0    # <1 = structurally under-supplied (tail-end, low pressure)
    tau_up_h: float = 0.0         # water age on arrival at this segment (trunk + storage), hours
    tau_local_h: float = 0.3      # extra water age inside the local segment at nominal flow
    base_turb: float = 0.6        # NTU
    base_tds: float = 285.0       # mg/L
    base_ph: float = 7.4
    equity: bool = True           # include in LPCD equity analysis
    meter: str = "YF-DN25 hall-effect"


@dataclass
class EventSpec:
    id: str
    type: str
    target: str                   # node id, group id ("G-<parent>") or "GW"
    start: datetime
    end: datetime
    params: dict = field(default_factory=dict)
    daily_window: tuple[int, int] | None = None   # active only between these hours each day
    description: str = ""
    expected_alerts: tuple[str, ...] = ()
    propagate: bool = False       # affects all nodes downstream of target

    def active(self, t: datetime) -> bool:
        if not (self.start <= t < self.end):
            return False
        if self.daily_window:
            h0, h1 = self.daily_window
            h = t.hour + t.minute / 60
            return h0 <= h < h1
        return True


@dataclass
class GatewaySpec:
    name: str
    lat: float
    lon: float
    antenna_gain_dbi: float = 5.0
    backhaul: str = "4G/GSM (SIM) with 72h local buffer"


@dataclass
class Group:
    id: str
    parent: str
    children: list[str]
    name: str


@dataclass
class Scenario:
    key: str
    title: str
    region: str
    start: datetime
    days: int
    step_min: int
    train_days: int
    nodes: list[NodeSpec]
    events: list[EventSpec]
    gateway: GatewaySpec
    seed: int = 7
    cl_dose: float = 1.8          # mg/L free chlorine leaving the WTP
    decay_k: float = 0.06         # bulk chlorine decay, 1/h at 25 C
    base_loss: dict[str, float] = field(default_factory=dict)
    contacts: dict[str, str] = field(default_factory=dict)
    role_alias: dict[str, str] = field(default_factory=dict)
    cloudiness: tuple[float, float] = (0.2, 0.75)   # daily solar availability range

    def __post_init__(self):
        self.node_map = {n.id: n for n in self.nodes}
        self.children: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            if n.parent:
                self.children[n.parent].append(n.id)
        self.groups: list[Group] = []
        for n in self.nodes:
            if self.children[n.id]:
                feeds_reservoirs = any(self.node_map[c].kind == "reservoir" for c in self.children[n.id])
                gname = ("Trunk main" if feeds_reservoirs else "Supply line") if n.kind == "source" else \
                    "Village distribution" if n.kind == "reservoir" else "Branch"
                self.groups.append(Group(f"G-{n.id}", n.id, list(self.children[n.id]),
                                         f"{gname}: {n.name}"))
        self.group_map = {g.id: g for g in self.groups}
        self._desc = {n.id: self._descendants(n.id) for n in self.nodes}

    # ------------------------------------------------------------------ helpers
    def _descendants(self, nid: str) -> list[str]:
        out = [nid]
        for c in self.children[nid]:
            out.extend(self._descendants(c))
        return out

    def descendants(self, nid: str) -> list[str]:
        return self._desc[nid]

    @property
    def steps(self) -> int:
        return int(self.days * 24 * 60 / self.step_min)

    @property
    def step(self) -> timedelta:
        return timedelta(minutes=self.step_min)

    @property
    def train_end(self) -> datetime:
        return self.start + timedelta(days=self.train_days)

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.days)

    def event_locations(self, ev: EventSpec) -> set[str]:
        if ev.target in self.node_map and ev.propagate:
            return set(self.descendants(ev.target))
        return {ev.target}

    def location_name(self, loc: str) -> str:
        if loc in self.node_map:
            return self.node_map[loc].name
        if loc in self.group_map:
            return self.group_map[loc].name
        if loc == "GW":
            return self.gateway.name
        return loc

    def village_of(self, loc: str) -> str:
        if loc in self.node_map:
            return self.node_map[loc].village
        if loc in self.group_map:
            return self.node_map[self.group_map[loc].parent].village
        return "All"


def _d(base: datetime, day: int, hh: int = 0, mm: int = 0) -> datetime:
    return base + timedelta(days=day, hours=hh, minutes=mm)


# =============================================================================
# Scenario 1: Mission Bhagiratha segment (mandal scale)
# =============================================================================
def mandal_scenario(seed: int = 7, days: int = 14) -> Scenario:
    s = datetime(2026, 7, 1)   # monsoon: peak season for water-borne disease
    WTP, A, B, C = "Mission Bhagiratha WTP", "Ramapuram", "Gollapalli", "Chintalapalle"
    nodes = [
        NodeSpec("N01", 1, "WTP clear-water outlet", WTP, "source", 18.0200, 78.7600,
                 tau_up_h=0.0, tau_local_h=0.1, base_turb=0.35, base_tds=270, base_ph=7.5,
                 equity=False, meter="DN150 clamp-on ultrasonic (pulse out)"),
        NodeSpec("N02", 2, "Ramapuram OHSR outlet", A, "reservoir", 18.0480, 78.7900, "N01",
                 tau_up_h=2.0, tau_local_h=4.0, base_turb=0.45, equity=False,
                 meter="DN80 Woltmann (pulse out)"),
        NodeSpec("N03", 3, "Ramapuram ZPHS school tap", A, "tap", 18.0530, 78.7960, "N02",
                 population=600, lpcd_target=45, profile="school", tau_up_h=6.0, tau_local_h=1.0,
                 base_turb=0.55, equity=False),
        NodeSpec("N04", 4, "Ramapuram main habitation", A, "tap", 18.0430, 78.7980, "N02",
                 population=1800, tau_up_h=6.0, tau_local_h=1.5, base_turb=0.6),
        NodeSpec("N05", 5, "Gollapalli OHSR outlet", B, "reservoir", 18.0600, 78.7450, "N01",
                 tau_up_h=2.5, tau_local_h=5.0, base_turb=0.45, equity=False,
                 meter="DN80 Woltmann (pulse out)"),
        NodeSpec("N06", 6, "Gollapalli main habitation", B, "tap", 18.0660, 78.7390, "N05",
                 population=2200, tau_up_h=7.5, tau_local_h=1.5, base_turb=0.6),
        NodeSpec("N07", 7, "Gollapalli ward 7 (tail-end)", B, "tap", 18.0690, 78.7530, "N05",
                 population=900, supply_factor=0.85, tau_up_h=7.5, tau_local_h=2.5, base_turb=0.7),
        NodeSpec("N08", 8, "Chintalapalle OHSR outlet", C, "reservoir", 17.9950, 78.8000, "N01",
                 tau_up_h=3.0, tau_local_h=5.0, base_turb=0.5, equity=False,
                 meter="DN80 Woltmann (pulse out)"),
        NodeSpec("N09", 9, "Chintalapalle main habitation", C, "tap", 17.9890, 78.8060, "N08",
                 population=1500, tau_up_h=8.0, tau_local_h=1.5, base_turb=0.65),
        NodeSpec("N10", 10, "Chintalapalle thanda (tail-end hamlet)", C, "tap", 17.9810, 78.8150, "N08",
                 population=650, supply_factor=0.55, tau_up_h=8.0, tau_local_h=3.0, base_turb=0.9),
    ]
    events = [
        EventSpec("E1", "leak", "G-N02", _d(s, 3, 3), _d(s, 5, 11), {"lpm": 25},
                  description="Burst joint on Ramapuram distribution line (~25 L/min)",
                  expected_alerts=("WATER_LOSS", "LEAK_CONFIRMED")),
        EventSpec("E2", "theft", "G-N01", _d(s, 5), _d(s, 12), {"lpm": 90}, daily_window=(18, 23),
                  description="Illegal tapping of trunk main by agricultural motor, 18:00-23:00 daily",
                  expected_alerts=("WATER_LOSS", "THEFT_SUSPECTED")),
        EventSpec("E3", "sensor_stuck", "N06", _d(s, 4, 8), _d(s, 5, 2), {"sensor": "ph"},
                  description="pH probe at Gollapalli main frozen (ADC lock-up)",
                  expected_alerts=("SENSOR_FAULT",)),
        EventSpec("E4", "contamination", "N10", _d(s, 7, 5), _d(s, 8, 14),
                  {"turb_add": 12.0, "cl_factor": 0.05, "tds_add": 80, "ph_shift": -0.25,
                   "attack_rate": 0.04},
                  description="Sewage ingress through cracked pipe near drain at thanda",
                  expected_alerts=("CONTAMINATION_RISK",)),
        EventSpec("E5", "gateway_outage", "GW", _d(s, 8, 10), _d(s, 8, 16),
                  description="Gateway 4G backhaul down (tower power cut); gateway buffers data",
                  expected_alerts=("GATEWAY_OFFLINE",)),
        EventSpec("E6", "node_down", "N07", _d(s, 9, 9), _d(s, 9, 17),
                  description="Node N07 solar cable cut (vandalism) - node silent",
                  expected_alerts=("NODE_OFFLINE",)),
        EventSpec("E7", "chlorination_failure", "N01", _d(s, 10, 6), _d(s, 10, 20), {"dose": 0.12,
                  "attack_rate": 0.002},
                  description="Chlorine dosing pump failure at WTP for 14 h",
                  expected_alerts=("SOURCE_CHLORINATION_FAILURE",)),
        EventSpec("E8", "groundwater_mixing", "N05", _d(s, 11, 8), _d(s, days), {"fraction": 0.45, "tds": 1600},
                  description="Panchayat pumps saline borewell water into Gollapalli OHSR", propagate=True,
                  expected_alerts=("HIGH_TDS",)),
        EventSpec("E9", "chronic_inequity", "N10", s, _d(s, days),
                  description="Thanda at tail-end gets ~55% of design supply (low pressure)",
                  expected_alerts=("EQUITY_SHORTFALL",)),
    ]
    contacts = {}
    for v in (A, B, C):
        contacts[f"gp_secretary:{v}"] = f"Panchayat Secretary, {v}"
        contacts[f"asha:{v}"] = f"ASHA worker, {v}"
        contacts[f"mb_ae:{v}"] = f"AE (Intra-village), Mission Bhagiratha - {v}"
    contacts.update({
        "phc_mo": "Medical Officer, PHC",
        "dmho": "District Medical & Health Officer",
        "mb_dee": "DEE (Grid), Mission Bhagiratha",
        "wtp_operator": "WTP shift operator",
        "technician": "JalNetra field technician",
        "mpdo": "Mandal Parishad Development Officer",
    })
    return Scenario(
        key="mandal", title="Mission Bhagiratha segment - 3 villages, 10 nodes",
        region="Siddipet district, Telangana (illustrative)", start=s, days=days, step_min=15,
        train_days=3, nodes=nodes, events=[e for e in events if e.start < _d(s, days)],
        gateway=GatewaySpec("LoRaWAN gateway on Ramapuram GP office", 18.0350, 78.7750),
        seed=seed, base_loss={"G-N01": 0.035, "G-N02": 0.05, "G-N05": 0.06, "G-N08": 0.07},
        contacts=contacts,
    )


# =============================================================================
# Scenario 2: School pilot (the 2-3 node physical prototype)
# =============================================================================
def pilot_scenario(seed: int = 11, days: int = 4) -> Scenario:
    s = datetime(2026, 12, 7)   # a Monday
    V = "ZPHS School (pilot)"
    nodes = [
        NodeSpec("P1", 21, "Sump / overhead-tank outlet", V, "source", 18.10200, 78.85200,
                 tau_up_h=0.0, tau_local_h=0.3, base_turb=0.5, base_tds=300, base_ph=7.4, equity=False),
        NodeSpec("P2", 22, "Drinking-water tap row", V, "tap", 18.10230, 78.85245, "P1",
                 population=450, lpcd_target=5, profile="school", tau_up_h=0.3, tau_local_h=0.4,
                 base_turb=0.55, equity=False),
        NodeSpec("P3", 23, "Mid-day-meal kitchen tap", V, "tap", 18.10175, 78.85250, "P1",
                 population=450, lpcd_target=4, profile="school", tau_up_h=0.3, tau_local_h=0.5,
                 base_turb=0.6, equity=False),
    ]
    events = [
        EventSpec("T1", "leak", "G-P1", _d(s, 1, 10, 30), _d(s, 1, 13, 30), {"lpm": 3.0},
                  description="Bench test: drain valve opened 3 L/min between meters",
                  expected_alerts=("WATER_LOSS",)),
        EventSpec("T2", "node_down", "P2", _d(s, 1, 15), _d(s, 1, 17),
                  description="Bench test: battery disconnected on P2",
                  expected_alerts=("NODE_OFFLINE",)),
        EventSpec("T3", "contamination", "P3", _d(s, 2, 10), _d(s, 2, 12, 30),
                  {"turb_add": 9.0, "cl_factor": 0.08, "tds_add": 40, "ph_shift": -0.1, "attack_rate": 0.0},
                  description="Bench test: kaolin + dechlorinated water in P3 test loop",
                  expected_alerts=("CONTAMINATION_RISK",)),
        EventSpec("T4", "chlorination_failure", "P1", _d(s, 3, 9), _d(s, 3, 13), {"dose": 0.05,
                  "attack_rate": 0.0},
                  description="Bench test: unchlorinated (dechlorinated) feed to sump",
                  expected_alerts=("SOURCE_CHLORINATION_FAILURE",)),
    ]
    roles = {"gp_secretary": "school_hm", "asha": "school_hm", "mb_ae": "mentor", "phc_mo": "school_hm",
             "mb_dee": "mentor", "wtp_operator": "mentor", "technician": "student_team",
             "dmho": "school_hm", "mpdo": "school_hm"}
    contacts = {"school_hm": "Headmaster", "mentor": "Project mentor (science teacher)",
                "student_team": "Student build team"}
    return Scenario(
        key="pilot", title="School pilot test-bed - 3 nodes",
        region="ZPHS school line, Telangana (illustrative)", start=s, days=days, step_min=15,
        train_days=1, nodes=nodes, events=[e for e in events if e.start < _d(s, days)],
        gateway=GatewaySpec("Single-channel gateway in school lab", 18.10210, 78.85215, 2.0,
                            "School Wi-Fi with 4G dongle fallback"),
        seed=seed, cl_dose=0.9, base_loss={"G-P1": 0.015}, contacts=contacts, role_alias=roles,
        cloudiness=(0.45, 0.9),
    )


SCENARIOS = {"mandal": mandal_scenario, "pilot": pilot_scenario}
