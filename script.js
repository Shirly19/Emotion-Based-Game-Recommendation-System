const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const bgMusic = document.getElementById("bgMusic");

let circles = [];
let score = 0;
let gameRunning = false;
let colors = ["#ff6b6b", "#feca57", "#1dd1a1", "#54a0ff", "#5f27cd"];

function createCircle() {
  const radius = 30;
  const x = Math.random() * (canvas.width - 2 * radius) + radius;
  const y = Math.random() * (canvas.height - 2 * radius) + radius;
  const color = colors[Math.floor(Math.random() * colors.length)];
  return { x, y, radius, color };
}

function drawCircle(c) {
  ctx.beginPath();
  ctx.arc(c.x, c.y, c.radius, 0, Math.PI * 2);
  ctx.fillStyle = c.color;
  ctx.fill();
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawGame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  circles.forEach(drawCircle);
  ctx.fillStyle = "#333";
  ctx.font = "20px Segoe UI";
  ctx.fillText("Score: " + score, 20, 30);
}

canvas.addEventListener("click", (e) => {
  if (!gameRunning) return;
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  for (let i = 0; i < circles.length; i++) {
    const c = circles[i];
    const dist = Math.hypot(mx - c.x, my - c.y);
    if (dist < c.radius) {
      score += 10;
      circles.splice(i, 1);
      circles.push(createCircle());
      break;
    }
  }
  drawGame();
});

function startGame() {
  gameRunning = true;
  circles = Array.from({ length: 5 }, createCircle);
  bgMusic.play();
  drawGame();
}

function resetGame() {
  gameRunning = false;
  score = 0;
  circles = [];
  bgMusic.pause();
  drawGame();
}

startBtn.onclick = startGame;
resetBtn.onclick = resetGame;

drawGame();
