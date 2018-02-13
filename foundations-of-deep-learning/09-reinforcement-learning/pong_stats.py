from collections import namedtuple
from pong_events import *

PONG_STATS_FIELDS = [
    "p1_points",
    "p2_points",
    "p1_bounces",
    "p2_bounces",
]

PongStats = namedtuple("PongStats", PONG_STATS_FIELDS)

def reduce(stats):
    p1_points = 0
    p2_points = 0
    p1_bounces = 0
    p2_bounces = 0

    num_games = len(stats)

    for stat in stats:
        p1_points += stat.p1_points
        p2_points += stat.p2_points
        p1_bounces += stat.p1_bounces
        p2_bounces += stat.p2_bounces

    return PongStats(
        p1_points = round(p1_points / num_games, 2),
        p2_points = round(p2_points / num_games, 2),
        p1_bounces = round(p1_bounces / num_games, 2),
        p2_bounces = round(p2_bounces / num_games, 2),
    )

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
