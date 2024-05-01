let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
ctx.strokeStyle = 'white';  // Set line color to white
ctx.lineWidth = 0.25;           // Set line thickness
let drawing = false;

function getEventPosition(event) {
    let rect = canvas.getBoundingClientRect();  // Gets the canvas position
    let x = event.clientX - rect.left;
    let y = event.clientY - rect.top;
    if (event.touches) {
        x = event.touches[0].clientX - rect.left;
        y = event.touches[0].clientY - rect.top;
    }
    return { x: x / 10, y: y / 10 };  // Adjust for scale
}

function startDrawing(event) {
    if (event.touches && event.touches[0].touchType !== 'stylus') return;  // Ignore finger touches
    drawing = true;
    const pos = getEventPosition(event);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
    event.preventDefault(); // Prevent scrolling and other unwanted gestures
}

function draw(event) {
    if (!drawing) return;
    const pos = getEventPosition(event);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    event.preventDefault();
}

function stopDrawing(event) {
    if (event.touches && event.touches[0].touchType !== 'stylus' && !drawing) return;
    drawing = false;
    ctx.closePath();
    event.preventDefault();
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing); // Optionally stop drawing when moving out

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchcancel', stopDrawing); // Handle interruptions

function clearCanvas() {
    ctx.fillStyle = 'black'; // Set the fill color to black
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill the canvas with black
}

function submitDrawing() {
    let image = canvas.toDataURL('image/png');
    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({'image_data': image})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('displayedImage').src = data.image_data;
        console.log('Image submitted and displayed.');
    })
    .catch(error => console.error('Error:', error));
}
