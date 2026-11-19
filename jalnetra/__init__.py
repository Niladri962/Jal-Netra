"""JalNetra - IoT virtual water grid simulator.

Simulates the full pipeline of the JalNetra system end to end:

    physical network  ->  sensor nodes  ->  LoRa radio  ->  gateway  ->  cloud backend
    (hydraulics +        (noise, faults,    (link budget,   (store &     (SQLite, AI engine,
     water quality)       solar power,       airtime,        forward)     alerts, dashboard)
                          17-byte payload)   packet loss)

Every "real-world" problem (leak, theft, contamination, chlorination failure,
sensor fault, outage ...) is injected as a ground-truth event so that the
detection layer can be scored objectively.
"""

__version__ = "1.0.0"
