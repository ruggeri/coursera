import aiohttp.web
import asyncio
import json
import pong
import pong_constants

def game_state_to_json(state):
    return json.dumps({
        "paddle1Pos": state.paddle1_pos,
        "paddle2Pos": state.paddle2_pos,
        "ballPos": list(state.ball_pos),
        "ballVel": list(state.ball_vel)
    })

async def index(request):
    with open('static/index.html') as f:
        return aiohttp.web.Response(
            text = f.read(), content_type = 'text/html'
        )

PLAY_LEARNED_AI = True
async def play_learned_ai(callback):
    import config
    import graph as g
    import main
    import play
    import tensorflow as tf
    with tf.Session() as session:
        graph = g.build_graph()
        saver = tf.train.Saver()
        saver.restore(session, config.CHECKPOINT_FILENAME)
        await play.async_evaluate_performance(
            session,
            graph,
            training_mode = True,
            callback = callback
        )

async def play_hand_ai(callback):
    game = pong.PongGame(training_mode = False)
    while True:
        game.play_default_move(pong_constants.PLAYER1)
        game.play_default_move(pong_constants.PLAYER2)
        game.evolve()
        await callback(game)

async def websocket_handler(request):
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)

    async def callback(game):
        ws.send_str(game_state_to_json(game.state))
        await asyncio.sleep(pong_constants.POLL_FREQUENCY)

    if PLAY_LEARNED_AI:
        await play_learned_ai(callback)
    else:
        await play_hand_ai(callback)

    return ws

app = aiohttp.web.Application()
app.router.add_get('/', index)
app.router.add_static('/static', 'static')
app.router.add_get("/websocket", websocket_handler)

if __name__ == '__main__':
    aiohttp.web.run_app(app, port = 8080)
