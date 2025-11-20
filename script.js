// --- 宇宙特效核心逻辑 ---
const canvas = document.getElementById('star-canvas');
const ctx = canvas.getContext('2d');

let width, height;
let particles = [];
let meteors = []; // 存储流星

function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
}
window.addEventListener('resize', resize);
resize();

// --- 星星类 (背景) ---
class Particle {
    constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.vx = (Math.random() - 0.5) * 0.2;
        this.vy = (Math.random() - 0.5) * 0.2;
        this.size = Math.random() * 2;
        this.color = `rgba(174, 194, 224, ${Math.random() * 0.5 + 0.3})`;
    }
    update() {
        this.x += this.vx;
        this.y += this.vy;
        if (this.x < 0) this.x = width;
        if (this.x > width) this.x = 0;
        if (this.y < 0) this.y = height;
        if (this.y > height) this.y = 0;
    }
    draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
    }
}

// --- 流星类 (新增) ---
class Meteor {
    constructor() {
        this.reset();
    }

    reset() {
        // 从右上角区域随机生成
        this.x = Math.random() * width + width * 0.5; 
        this.y = Math.random() * height * -0.5; 
        this.len = Math.random() * 80 + 10; // 尾巴长度
        this.speed = Math.random() * 10 + 6; // 速度
        this.size = Math.random() * 1 + 0.5;
        this.active = false; // 默认不显示
    }

    activate() {
        this.active = true;
    }

    update() {
        if (!this.active) return;
        
        // 向左下方移动
        this.x -= this.speed;
        this.y += this.speed;

        // 如果移出屏幕左下角，重置
        if (this.x < -this.len || this.y > height + this.len) {
            this.active = false;
            this.reset();
        }
    }

    draw() {
        if (!this.active) return;
        
        ctx.beginPath();
        // 创建渐变尾巴
        let gradient = ctx.createLinearGradient(this.x, this.y, this.x + this.len, this.y - this.len);
        gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
        gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = this.size;
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(this.x + this.len, this.y - this.len);
        ctx.stroke();
    }
}

// 初始化
function init() {
    particles = [];
    meteors = [];
    // 创建星星
    for (let i = 0; i < Math.floor(window.innerWidth / 10); i++) {
        particles.push(new Particle());
    }
    // 创建流星池 (虽然只显示几个，但准备好对象)
    for (let i = 0; i < 5; i++) {
        meteors.push(new Meteor());
    }
}
init();

// 鼠标交互
let mouse = { x: null, y: null };
window.addEventListener('mousemove', (e) => { mouse.x = e.x; mouse.y = e.y; });
window.addEventListener('mouseout', () => { mouse.x = null; mouse.y = null; });

// 动画循环
function animate() {
    ctx.clearRect(0, 0, width, height);

    // 1. 绘制星星和连线
    particles.forEach((p, index) => {
        p.update();
        p.draw();
        // 鼠标连线逻辑
        if (mouse.x != null) {
            const dx = p.x - mouse.x;
            const dy = p.y - mouse.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < 150) {
                ctx.beginPath();
                ctx.strokeStyle = `rgba(0, 198, 255, ${1 - distance/150})`;
                ctx.lineWidth = 1;
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(mouse.x, mouse.y);
                ctx.stroke();
            }
        }
    });

    // 2. 绘制流星
    meteors.forEach(m => {
        m.update();
        m.draw();
    });
    
    // 随机触发流星 (约 1% 的概率每一帧触发一个，如果没有正在飞的)
    if (Math.random() < 0.01) {
        const inactiveMeteor = meteors.find(m => !m.active);
        if (inactiveMeteor) inactiveMeteor.activate();
    }

    requestAnimationFrame(animate);
}
animate();

// --- 打字机逻辑 (仅在首页需要) ---
const typingEl = document.getElementById('typing-text');
if(typingEl) {
    const text = typingEl.getAttribute('data-text') || "Welcome to my universe.";
    let index = 0;
    function typeWriter() {
        if (index < text.length) {
            typingEl.innerHTML += text.charAt(index);
            index++;
            setTimeout(typeWriter, 100);
        }
    }
    typeWriter();
}