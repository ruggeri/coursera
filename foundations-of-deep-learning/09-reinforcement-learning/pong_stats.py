from collections import namedtuple
from pong_events import *

PONG_STATS_FIELDS = [
    "p1_points",
    "p2_points",
    "p1_bounces",
    "p2_bounces",
]

PongStats = namedtuple("PongStats", PONG_STATS_FIELDS)

def new():
    return PongStats(
        p1_points = 0,
        p2_points = 0,
        p1_bounces = 0,
        p2_bounces = 0,
    )

def add_events(stats, events):
    return PongStats(
        p1_points = stats.p1_points + events.count(P1_POINT_SCORED),
        p2_points = stats.p2_points + events.count(P2_POINT_SCORED),
        p1_bounces = stats.p1_bounces + events.count(P1_BOUNCE),
        p2_bounces = stats.p2_bounces + events.count(P2_BOUNCE),
    )

def score(stats):
    return (stats.p1_points - stats.p2_points)

def total_points(stats):
    return (stats.p1_points + stats.p2_points)
