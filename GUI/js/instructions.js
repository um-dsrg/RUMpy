const videoAnchor = document.getElementById('videoAnchor');
const videoFrame = document.getElementById('videoEmbed');

function showVideo() {
    videoFrame.style.visibility = 'visible';
    videoFrame.style.display = '';
}

videoAnchor.addEventListener('click', showVideo);
