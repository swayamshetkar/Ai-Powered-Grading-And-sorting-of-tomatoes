const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Ask for camera access
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

function sendFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL("image/jpeg");

    fetch("/detect_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL })
    })
    .then(res => res.json())
    .then(data => {
        if(data.image){
            const img = new Image();
            img.src = data.image;
            img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }
    })
    .catch(console.error)
    .finally(() => setTimeout(sendFrame, 200)); // ~5 FPS
}

// Start loop after video is ready
video.addEventListener("loadeddata", sendFrame);
