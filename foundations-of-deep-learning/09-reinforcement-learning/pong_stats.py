from collections import namedtuple

PONG_STATS_FIELDS = [
    "p1_points",
    "p2_points",
    "p1_bounces",
    "p2_bounces",
    "p1_nudge_ups",
    "p1_nudge_downs",
    "p2_nudge_ups",
    "p1_nudge_downs",
]

PongStats = namedtuple("PongStats", PONG_STATS_FIELDS)

def new():
    return PongStats(
        p1_points = 0,
        p2_points = 0,
        p1_bounces = 0,
        p2_bounces = 0,
        p1_nudge_ups = 0,
        p1_nudge_downs = 0,
        p2_nudge_ups = 0,
        p2_nudge_downs = 0,
    )

def score(stats):
    return (stats.p1_points - stats.p2_points)
