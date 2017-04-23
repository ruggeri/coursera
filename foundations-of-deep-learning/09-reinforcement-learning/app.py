import aiohttp.web
import asyncio
from collections import namedtuple
import json
import pong

POLL_FREQUENCY = 0.02

GAMES = {}

app = aiohttp.web.Application()

JSONState = namedtuple("JSONState", [
    "paddle1Pos",
    "paddle2Pos",
    "ballPos",
    "ballVel",
    "score",
])

def game_state_to_json(state):
    return json.dumps({
        "paddle1Pos": state.paddle1_pos,
        "paddle2Pos": state.paddle2_pos,
        "ballPos": list(state.ball_pos),
        "ballVel": list(state.ball_vel),
        "score": state.score
    })

async def index(request):
    with open('static/index.html') as f:
        return aiohttp.web.Response(
            text = f.read(), content_type='text/html'
        )

async def websocket_handler(request):
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)

    game = pong.PongGame()
    while True:
        game.play_computer_move(pong.PLAYER1)
        game.play_computer_move(pong.PLAYER2)
        game.step()
        ws.send_str(game_state_to_json(game.state()))
        await asyncio.sleep(POLL_FREQUENCY)

    return ws

async def test_websocket_handler(request):
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)

    async def cb(game):
        ws.send_str(game_state_to_json(game.state()))
        await asyncio.sleep(POLL_FREQUENCY)
    import main
    await main.play_example_game(cb, training_mode = True)

    return ws

app.router.add_static('/static', 'static')
app.router.add_get('/', index)
app.router.add_get("/websocket", test_websocket_handler)

if __name__ == '__main__':
    aiohttp.web.run_app(app, port = 8080)
