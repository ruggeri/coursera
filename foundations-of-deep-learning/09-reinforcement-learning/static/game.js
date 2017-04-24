const CANVAS_DIM = 400;
const BALL_RADIUS = 0.025;
const PADDLE_SIZE = 0.10;
const PADDLE_DEPTH = 0.02;
const PLAYER1 = 1;
const PLAYER2 = 2;

const canvasEl = document.getElementById("game");
const ctx = canvasEl.getContext("2d");

function drawGame(msg) {
  msg = JSON.parse(msg);

  ctx.fillStyle = "green";
  ctx.fillRect(0, 0, CANVAS_DIM, CANVAS_DIM);

  drawBall(msg.state.ballPos);
  drawPaddle(msg.state.paddle1Pos, PLAYER1);
  drawPaddle(msg.state.paddle2Pos, PLAYER2);
  drawScore(msg.stats.score);
}

function drawBall(pos) {
  let [x, y] = [pos[1] * CANVAS_DIM, pos[0] * CANVAS_DIM];

  ctx.fillStyle = "red";
  ctx.beginPath();
  ctx.arc(x, y, BALL_RADIUS * CANVAS_DIM, 0, 2 * Math.PI);
  ctx.fill();
}

function drawPaddle(pos, playernum) {
  let x_min;
  if (playernum == PLAYER1) {
    x_min = 0;
  } else if (playernum == PLAYER2) {
    x_min = (1.0 - PADDLE_DEPTH) * CANVAS_DIM;
  }
  let y_min = CANVAS_DIM * (pos - (PADDLE_SIZE / 2));

  ctx.fillStyle = "blue";
  ctx.fillRect(
    x_min,
    y_min,
    PADDLE_DEPTH * CANVAS_DIM,
    PADDLE_SIZE * CANVAS_DIM
  );
}

function drawScore(score) {
  const el = document.getElementById("score");
  el.innerHTML = `Score: ${score}`;
}

let exampleSocket = new WebSocket(
  "ws://localhost:8080/websocket"
);

exampleSocket.onopen = () => {
  console.log("we connected!");
};

exampleSocket.onmessage = (msg) => { drawGame(msg.data) };
