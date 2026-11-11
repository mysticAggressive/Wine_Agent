"""Project-wide constants. See DEVELOPMENT_PLAN.md section 3."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CENTER = (40.4168, -3.7038)  # Puerta del Sol (lat, lon)
RADIUS_M = 4000
TIMEZONE = "Europe/Madrid"

# Price at a venue = retail price x markup for the venue kind.
MARKUP = {"shop": 1.0, "bar": 2.0, "restaurant": 2.5}

MIN_PAIRING_SCORE = 0.5
MAX_PER_VENUE = 2
WALK_SPEED_M_PER_MIN = 80
FEATURES = ("taste", "quality", "price_fit", "pairing", "distance")

# Starting points offered in the preference panel: label -> (lat, lon).
LANDMARKS = {
    "Puerta del Sol": (40.4168, -3.7038),
    "Plaza Mayor": (40.4155, -3.7074),
    "Atocha": (40.4066, -3.6892),
    "Retiro": (40.4153, -3.6845),
    "Gran Vía – Callao": (40.4200, -3.7058),
    "Malasaña – Plaza Dos de Mayo": (40.4268, -3.7040),
    "Chamberí – Plaza de Olavide": (40.4327, -3.7010),
    "Lavapiés": (40.4089, -3.7007),
}
