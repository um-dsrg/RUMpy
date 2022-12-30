// Video Player Source: https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Client-side_web_APIs/Video_and_audio_APIs

// Video/image input
const content = document.querySelector('input')

// Metadata input and div
const metadataInput = document.getElementById('metadata')
const metadataDiv = document.getElementById('metadataLoad')

// Custom video player controls and seek bar
const videoDiv = document.querySelector('.videoplayer')
const media = document.querySelector('video');
const controls = document.querySelector('.controls');

const play = document.querySelector('.play');
const stop = document.querySelector('.stop');
const rwd = document.querySelector('.rwd');
const fwd = document.querySelector('.fwd');
const frameBck = document.querySelector('.frameBck');
const frameFwd = document.querySelector('.frameFwd');

const frameCut = document.querySelector('.frameCut');

const timerWrapper = document.querySelector('.timer');
const timer = document.querySelector('.timertext');
const timerBar = document.querySelector('.timer div');

// Cropping controls
const chosenFrameCanvas = document.getElementById('chosenFrameCanvas');
const chosenFrameDiv = document.getElementById('chosenFrame');

const crop = document.querySelector('.crop');
const cancelcrop = document.querySelector('.cancelcrop');
const save = document.querySelector('.save');

// Cropped image display
const croppedFrameCanvas = document.getElementById('croppedFrameCanvas');
const croppedFrameDiv = document.getElementById('croppedFrame');

// Super-resolution controls
const superResolvedFrameCanvas = document.getElementById('superResolvedFrameCanvas');
const superResolvedFrameDiv = document.getElementById('superResolvedFrame');

const superResolve = document.querySelector('.superresolve4');
const downloadSR = document.getElementById('downloadSR');
const showComp = document.getElementById('showComp');

// Comparison slider and canvases
const comparisonFrameDiv = document.getElementById('comparisonFrame');
const sliderCanvas = document.getElementById('sliderCanvas');

const sliderContext = sliderCanvas.getContext('2d');
sliderContext.imageSmoothingEnabled = false;

const comparisonCanvasBefore = document.getElementById('comparisonCanvasBefore');
const comparisonCanvasAfter = document.getElementById('comparisonCanvasAfter');
const downloadComp = document.getElementById('downloadComp');

// Font-Awesome icons for pause and play
const playIcon = '\uf04b';
const pauseIcon = '\uf04c';

// Option to input metadata with image
var useMetadata;

// Boolean to store whether video or not
var isVideo;

document.body.scrollTop = 0;
document.documentElement.scrollTop = 0;

// Load the input from the user and display the corresponding div
function loadContent(event) {
    // Reset the metadata file
    metadataInput.value = '';

    // Regex expression to get the file extension
    var re = /(?:\.([^.]+))?$/;

    var fileType = re.exec(content.value)[1];

    // Check if the file is an image or a video
    if (fileType == 'png' || fileType == 'JPG' || fileType == 'jpg' || fileType == 'jpeg' || fileType == 'gif' || fileType == 'tif' || fileType == 'tiff') {
        isVideo = false;

        var image = new Image();
        image.src = URL.createObjectURL(content.files[0]);

        var file = event.target.files[0];

        // Check if the file is a tif image since these are generally not supported natively
        // If the file is a tif image, open it with the FileReader, use the tiff.js library and draw on the canvas
        // Else just draw the image directly on the canvas
        if (fileType == 'tif' || fileType == 'tiff') {
            // Source: https://github.com/seikichi/tiff.js/tree/master
            var reader = new FileReader();

            reader.onload = function () {
                return function (e) {
                    // Get the tiff image and store it as a canvas
                    var buffer = e.target.result;
                    var tiff = new Tiff({buffer: buffer});
                    var tiffCanvas = tiff.toCanvas();
                    var width = tiff.width();
                    var height = tiff.height();

                    if (tiffCanvas) {
                        var context = chosenFrameCanvas.getContext('2d');
                        chosenFrameCanvas.width = width;
                        chosenFrameCanvas.height = height;
                        context.imageSmoothingEnabled = false;
                        context.drawImage(tiffCanvas, 0, 0, chosenFrameCanvas.width, chosenFrameCanvas.height);
                    }
                };
            }(file);

            reader.readAsArrayBuffer(file);
        } else {
            image.onload = function () {
                var context = chosenFrameCanvas.getContext('2d');
                chosenFrameCanvas.width = image.width;
                chosenFrameCanvas.height = image.height;
                context.imageSmoothingEnabled = false;
                context.drawImage(image, 0, 0, chosenFrameCanvas.width, chosenFrameCanvas.height);
            }
        }

        // Hide all the divs except the frame canvas every time an image is loaded
        // Setting the display to 'none' removes the block from the flow of the page
        videoDiv.style.display = 'none';
        controls.style.display = 'none';

        videoDiv.style.visibility = 'hidden';
        controls.style.visibility = 'hidden';

        croppedFrameCanvas.style.visibility = 'hidden';
        croppedFrameDiv.style.visibility = 'hidden';

        superResolvedFrameCanvas.style.visibility = 'hidden';
        superResolvedFrameDiv.style.visibility = 'hidden';

        comparisonCanvasBefore.style.visibility = 'hidden';
        comparisonCanvasAfter.style.visibility = 'hidden';
        sliderCanvas.style.visibility = 'hidden';
        comparisonFrameDiv.style.visibility = 'hidden';

        if (useMetadata == false){
            chosenFrameCanvas.style.visibility = 'visible';
            chosenFrameDiv.style.visibility = 'visible';

            metadataDiv.style.display = 'none';
            metadataDiv.style.visibility = 'hidden';

            chosenFrameDiv.scrollIntoView();
        } else {
            chosenFrameCanvas.style.visibility = 'hidden';
            chosenFrameDiv.style.visibility = 'hidden';

            metadataDiv.style.display = '';
            metadataDiv.style.visibility = 'visible';

            metadataDiv.scrollIntoView();
        }
    } else {
        isVideo = true;

        var videoSource = document.getElementById('source');
        media.src = URL.createObjectURL(content.files[0]);
        media.type = 'video/' + fileType;
        media.appendChild(videoSource)

        // Hide all the divs except the video player every time a video is loaded
        // Setting the display to '' adds the block to the page flow
        chosenFrameCanvas.style.visibility = 'hidden';
        chosenFrameDiv.style.visibility = 'hidden';

        croppedFrameCanvas.style.visibility = 'hidden';
        croppedFrameDiv.style.visibility = 'hidden';

        superResolvedFrameCanvas.style.visibility = 'hidden';
        superResolvedFrameDiv.style.visibility = 'hidden';

        comparisonCanvasBefore.style.visibility = 'hidden';
        comparisonCanvasAfter.style.visibility = 'hidden';
        sliderCanvas.style.visibility = 'hidden';
        comparisonFrameDiv.style.visibility = 'hidden';

        media.currentTime = 0;
        timerBar.style.width = '0px';

        if (useMetadata == false){
            videoDiv.style.display = '';
            controls.style.display = '';

            videoDiv.style.visibility = 'visible';
            controls.style.visibility = 'visible';

            metadataDiv.style.display = 'none';
            metadataDiv.style.visibility = 'hidden';

            videoDiv.scrollIntoView();
        } else {
            videoDiv.style.display = 'none';
            controls.style.display = 'none';

            videoDiv.style.visibility = 'hidden';
            controls.style.visibility = 'hidden';

            metadataDiv.style.display = '';
            metadataDiv.style.visibility = 'visible';

            metadataDiv.scrollIntoView();
        }
    }
}

// Loaded metadata file
var metadataFileContents;

// Read the content of the metadata file and move to the next section
function loadedMetadata(event) {
    var file = event.target.files[0];

    var reader = new FileReader();

    reader.onload = function () {
        metadataFileContents = reader.result
    };

    reader.readAsText(file);

    if (isVideo == true) {
        videoDiv.style.display = '';
        controls.style.display = '';

        videoDiv.style.visibility = 'visible';
        controls.style.visibility = 'visible';

        videoDiv.scrollIntoView();
    } else {
        chosenFrameCanvas.style.visibility = 'visible';
        chosenFrameDiv.style.visibility = 'visible';

        chosenFrameDiv.scrollIntoView();
    }
}

// Handler for the play/pause button which stops any fastforwarding and rewinding, and toggles the play/pause
function playPauseMedia() {
    rwd.classList.remove('active');
    fwd.classList.remove('active');
    clearInterval(intervalRwd);
    clearInterval(intervalFwd);

    // Toggle the play/pause button
    if (media.paused) {
        play.setAttribute('data-icon', pauseIcon);
        media.play();
    } else {
        play.setAttribute('data-icon', playIcon);
        media.pause();
    }
}

// Handler for the stop button
function stopMedia() {
    media.pause();
    media.currentTime = 0;
    play.setAttribute('data-icon', playIcon);

    rwd.classList.remove('active');
    fwd.classList.remove('active');
    clearInterval(intervalRwd);
    clearInterval(intervalFwd);
}

var intervalFwd;
var intervalRwd;

// Handler for the rewind button which uses an interval of 200ms to wind the video backward
function mediaBackward() {
    clearInterval(intervalFwd);
    fwd.classList.remove('active');

    // Toggle between rewind and play
    if (rwd.classList.contains('active')) {
        rwd.classList.remove('active');
        clearInterval(intervalRwd);
        media.play();
    } else {
        rwd.classList.add('active');
        media.pause();
        intervalRwd = setInterval(windBackward, 200);
    }
}

// Handler for the fastforward button which uses an interval of 200ms to wind the video forward
function mediaForward() {
    clearInterval(intervalRwd);
    rwd.classList.remove('active');

    // Toggle between fastforward and play
    if (fwd.classList.contains('active')) {
        fwd.classList.remove('active');
        clearInterval(intervalFwd);
        media.play();
    } else {
        fwd.classList.add('active');
        media.pause();
        intervalFwd = setInterval(windForward, 200);
    }
}

var fastForwardRewindTime = 2;

// Function that winds back the video and is called from the rewind handler every 200ms
function windBackward() {
    if (media.currentTime <= fastForwardRewindTime) {
        rwd.classList.remove('active');
        clearInterval(intervalRwd);
        stopMedia();
    } else {
        media.currentTime -= fastForwardRewindTime;
    }
}

// Function that winds forward the video and is called from the fastforward handler every 200ms
function windForward() {
    if (media.currentTime >= media.duration - fastForwardRewindTime) {
        fwd.classList.remove('active');
        clearInterval(intervalFwd);
        stopMedia();
    } else {
        media.currentTime += fastForwardRewindTime;
    }
}

// This time is an estimate for 1 frame at 30fps (can be adjusted by user)
var frameTime;

// Handler for the frame backward button
function stepBackward() {
    if (media.currentTime <= frameTime) {
        stopMedia();
    } else {
        media.currentTime -= frameTime;
    }
}

// Handler for the frame forward function
function stepForward() {
    if (media.currentTime >= media.duration - frameTime) {
        stopMedia();
    } else {
        media.currentTime += frameTime;
    }
}

// Function that is called every time the video sends a 'timeupdate' event
// Updates the timer and the seeker bar in the video player
function setTime() {
    var hours = Math.floor(media.currentTime / 3600);
    var minutes = Math.floor((media.currentTime - (hours * 3600)) / 60);
    var seconds = Math.floor(media.currentTime - (hours * 3600) - (minutes * 60));
    var hourValue;
    var minuteValue;
    var secondValue;

    if (hours < 10) {
        hourValue = '0' + hours;
    } else {
        hourValue = hours;
    }

    if (minutes < 10) {
        minuteValue = '0' + minutes;
    } else {
        minuteValue = minutes;
    }

    if (seconds < 10) {
        secondValue = '0' + seconds;
    } else {
        secondValue = seconds;
    }

    var mediaTime = hourValue + ':' + minuteValue + ':' + secondValue;
    timer.textContent = mediaTime;

    var barLength = timerWrapper.clientWidth * (media.currentTime / media.duration);
    timerBar.style.width = barLength + 'px';
}

// Handler for the frame cut button
function chooseFrame() {
    var context = chosenFrameCanvas.getContext('2d');
    chosenFrameCanvas.width = media.videoWidth;
    chosenFrameCanvas.height = media.videoHeight;
    context.imageSmoothingEnabled = false;
    context.drawImage(media, 0, 0, chosenFrameCanvas.width, chosenFrameCanvas.height);

    chosenFrameCanvas.style.visibility = 'visible';
    chosenFrameDiv.style.visibility = 'visible';

    chosenFrameDiv.scrollIntoView();
}

// These next 5 mouse functions allow the user to go through the video using the seeker bar
var documentMouseDown;

function moveSlider(e, click) {
    if (documentMouseDown || click) {
        var currentX = e.x;
        var currentY = e.y;

        var timerRect = timerWrapper.getBoundingClientRect()

        if (currentX > timerRect.left && currentX < timerRect.right) {
            if (currentY < timerRect.bottom && currentY > timerRect.top) {
                var fullBar = timerRect.right - timerRect.left;
                var clickPercent = (currentX - timerRect.left) / fullBar;

                media.currentTime = media.duration * clickPercent;

                var barLength = timerWrapper.clientWidth * (media.currentTime / media.duration);
                timerBar.style.width = barLength + 'px';
            }
        }
    }
}

document.onmousedown = function(e) {
    documentMouseDown = true;
}

document.onmouseup = function(e) {
    documentMouseDown = false;
}

document.onmousemove = function(e) {
    moveSlider(e, false);
}

document.onclick = function(e) {
    moveSlider(e, true);
}

// The next 3 functions use the CropperJS library to display a cropping interface on the canvas
// and allow the use to start, cancel and save a crop

var cropper;
var cropperX = 0.0;
var cropperY = 0.0;
var cropperWidth = 0.0;
var cropperHeight = 0.0;
var cropping = false;

function startCropping() {
    if (cropping == false) {
        cropper = new Cropper(chosenFrameCanvas, {viewMode: 1, zoomable: false, scalable: false });
        cropping = true;
    } else {
        cropper.clear();
        cropper.destroy();
        cropping = false;
    }
}

function cancelCropping() {
    if (cropping == true) {
        cropper.clear();
        cropper.destroy();
        cropping = false;
    }
}

function saveCrop() {
    if (cropping == true) {
        var context = croppedFrameCanvas.getContext('2d');
        var croppedImage = new Image();

        croppedImage.onload = function() {
            croppedFrameCanvas.width = croppedImage.width;
            croppedFrameCanvas.height = croppedImage.height;
            context.imageSmoothingEnabled = false;
            context.drawImage(croppedImage, 0, 0, croppedFrameCanvas.width, croppedFrameCanvas.height);
        };

        var cropData = cropper.getData();
        cropperX = cropData['x'];
        cropperY = cropData['y'];
        cropperWidth = cropData['width'];
        cropperHeight = cropData['height'];

        var tempCropped = cropper.getCroppedCanvas();
        croppedImage.src = tempCropped.toDataURL('image/png');

        croppedFrameCanvas.style.visibility = 'visible';
        croppedFrameDiv.style.visibility = 'visible';

        cropper.clear();
        cropper.destroy();
        cropping = false;

        croppedFrameDiv.scrollIntoView();
    }
}

// Function to get the index of the nth occurence of a substring in a string
function nthIndex(inputString, pattern, n){
    var index = -1;
    while(n > 0 && index < inputString.length){
        n--;
        index++;

        index = inputString.indexOf(pattern, index);

        if (index < 0) {
            break;
        }
    }
    return index;
}

var apiLink;

// Carry out super-resolution on the cropped frame by sending a base64 string of the image to the API
// If a meta-attention model is used, the metadata from the csv file will also be sent as a regular string
function superResolveFrame() {
    // This is to super-resolve from the cropped canvas itself
    // var chosenImage = croppedFrameCanvas.toDataURL();

    // This is to super-resolve from the original canvas and do cropping on the API side
    var chosenImage = chosenFrameCanvas.toDataURL();

    // Convert image to base64
    var base64Img = chosenImage.substr(22);

    var requestDict = {'image': base64Img, 'x': cropperX, 'y': cropperY, 'width': cropperWidth, 'height': cropperHeight};

    if (useMetadata == true) {
        var metadataLines = metadataFileContents.split(/\r\n|\r|\n/g);
        var metadataHeader = metadataLines[0];

        var filename = content.files[0].name;

        var metadataRow;

        // This might be a little too hard-coded to parse a CSV file but it works for now.
        for (var i = 0; i < metadataLines.length; i++) {
            if (metadataLines[i].indexOf(filename) >= 0) {
                metadataRow = metadataLines[i];
                break;
            }
        }

        var metadataFirst = nthIndex(metadataRow, "\"", 1);
        var metadataSecond = nthIndex(metadataRow, "\"", 2);
        var metadataThird = nthIndex(metadataRow, "\"", 3);
        var metadataFourth = nthIndex(metadataRow, "\"", 4);

        var blur_kernel;
        var unmodified_blur_kernel;

        // So far mainly tested with blur_kernel only, changes in the API would need to be done for the unmodified_blur_kernel
        if (metadataHeader.includes('unmodified_blur_kernel') == true || (metadataThird >= 0 && metadataFourth >= 0)) {
            blur_kernel = metadataRow.substring(metadataFirst + 2, metadataSecond - 1);
            unmodified_blur_kernel = metadataRow.substring(metadataThird + 2, metadataFourth - 1);
            requestDict['blur_kernel'] = blur_kernel;
            requestDict['unmodified_blur_kernel'] = unmodified_blur_kernel;
        } else {
            if (metadataHeader.includes('blur_kernel') == true && metadataHeader.includes('QPI') == true) {
                blur_kernel = metadataRow.substring(metadataFirst + 2, metadataSecond - 1);
                qpi = metadataRow.substring(metadataSecond + 2);
                requestDict['blur_kernel'] = blur_kernel;
                requestDict['QPI'] = qpi;
            } else if (metadataHeader.includes('blur_kernel') == true){
                blur_kernel = metadataRow.substring(metadataFirst + 2, metadataSecond - 1);
                requestDict['blur_kernel'] = blur_kernel;
            }
        }
    }

    // $.post(apiLink + 'super_resolve',  requestDict).done(function(data) {
    $.post(apiLink + 'super_resolve_and_crop',  requestDict).done(function(data) {
        var context = superResolvedFrameCanvas.getContext('2d');
        var superResolvedImage = new Image();

        superResolvedImage.onload = function() {
            superResolvedFrameCanvas.width = superResolvedImage.width;
            superResolvedFrameCanvas.height = superResolvedImage.height;
            sliderCanvas.width = superResolvedImage.width;
            context.imageSmoothingEnabled = false;
            context.drawImage(superResolvedImage, 0, 0, superResolvedFrameCanvas.width, superResolvedFrameCanvas.height);

            sliderCanvas.height = (superResolvedFrameCanvas.height / superResolvedFrameCanvas.width) * sliderCanvas.width;
            sliderContext.imageSmoothingEnabled = false;
            sliderContext.drawImage(superResolvedImage, 0, 0, sliderCanvas.width, sliderCanvas.height);
            drawSliderImages(0.5);

            var afterContext = comparisonCanvasAfter.getContext('2d');
            comparisonCanvasAfter.width = superResolvedFrameCanvas.width;
            comparisonCanvasAfter.height = superResolvedFrameCanvas.height;
            afterContext.imageSmoothingEnabled = false;
            afterContext.drawImage(superResolvedFrameCanvas, 0, 0, comparisonCanvasAfter.width, comparisonCanvasAfter.height);
        }

        superResolvedImage.src = 'data:image/png;base64,'.concat(data);
    });

    // $.post(apiLink + 'super_resolve_bicubic',  requestDict).done(function(data) {
    $.post(apiLink + 'super_resolve_bicubic_and_crop',  requestDict).done(function(data) {
        var beforeContext = comparisonCanvasBefore.getContext('2d');
        var bicubicImage = new Image();

        bicubicImage.onload = function() {
            comparisonCanvasBefore.width = bicubicImage.width;
            comparisonCanvasBefore.height = bicubicImage.height;
            beforeContext.imageSmoothingEnabled = false;
            beforeContext.drawImage(bicubicImage, 0, 0, comparisonCanvasBefore.width, comparisonCanvasBefore.height);
            drawSliderImages(0.5);
        }

        bicubicImage.src = 'data:image/png;base64,'.concat(data);

        superResolvedFrameCanvas.style.visibility = 'visible';
        superResolvedFrameDiv.style.visibility = 'visible';

        superResolvedFrameDiv.scrollIntoView();
    });
}

// These next 5 mouse functions allow the user to go through the comparison image and slide from original to super-resolved
var canvasSliderMouseDown;

function moveImageSlider(e, click) {
    if (canvasSliderMouseDown || click) {
        var rect = sliderCanvas.getBoundingClientRect();

        var scaleX = sliderCanvas.width / rect.width;
        var currentX = (e.clientX - rect.left) * scaleX;

        sliderDivide = currentX / sliderCanvas.width;

        drawSliderImages(sliderDivide);
    }
}

sliderCanvas.onmousedown = function(e) {
    canvasSliderMouseDown = true;
}

sliderCanvas.onmouseup = function(e) {
    canvasSliderMouseDown = false;
}

sliderCanvas.onmousemove = function(e) {
    moveImageSlider(e, false);
}

sliderCanvas.onclick = function(e) {
    moveImageSlider(e, true);
}

// Function that loads the canvases (or any data url) as an image
function createSliderImage(source) {
    var loaded = false;
    var image = new Image();

    // TODO: maybe see how you can make this better???
    image.onload = function() {
        loaded = true;
    }

    image.src = source;

    return image;
}

// Function that draws a slider and splits a canvas into 2 images, before image on the left, after image on the right
// Source adapted from: https://gist.github.com/hongymagic/2403518 and http://jsfiddle.net/WkM5z/215/
function drawSliderImages(sliderDivide) {
    var after = createSliderImage(superResolvedFrameCanvas.toDataURL());
    var before = createSliderImage(comparisonCanvasBefore.toDataURL());

    var split = sliderDivide * sliderCanvas.width;
    var split_image = sliderDivide * before.width;

    sliderContext.drawImage(after, 0, 0, sliderCanvas.width, sliderCanvas.height);
    sliderContext.drawImage(before, 0, 0, split_image, before.height, 0, 0, split, sliderCanvas.height);

    sliderContext.fillStyle = "rgb(220, 50, 50)";
    sliderContext.fillRect(split - 2, 0, 4, sliderCanvas.height);
}

var downloadNameSR = 'image.png';

// Handler for the download button which gives the option to save the super-resolved image
function downloadSuperResolvedFrame() {
    var downloadLink = document.createElement('a');
    var downloadImage = document.getElementById('superResolvedFrameCanvas');

    downloadLink.href = downloadImage.toDataURL();
    downloadLink.download = downloadNameSR;

    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}

// Shwo the images with the slider and the stacked images
function showComparisonCanvases() {
    comparisonCanvasBefore.style.visibility = 'visible';
    comparisonCanvasAfter.style.visibility = 'visible';
    sliderCanvas.style.visibility = 'visible';
    comparisonFrameDiv.style.visibility = 'visible';

    comparisonFrameDiv.scrollIntoView();
}

var downloadNameComp = 'comparison.pdf';

// TODO: Check out how to show the bicubic image
// TODO: Check out how images are rendered on canvas
// TODO: Check out these links:
// https://developer.mozilla.org/en-US/docs/Web/CSS/image-rendering
// https://stackoverflow.com/questions/2670084/canvas-image-smoothing
// http://help.dottoro.com/lcasdhhx.php
// https://stackoverflow.com/questions/5403955/image-interpolation-mode-in-chrome-safari
// http://vaughnroyko.com/state-of-nearest-neighbor-interpolation-in-canvas/
// https://stackoverflow.com/questions/7615009/disable-interpolation-when-scaling-a-canvas

// Add either stacked or side-by-side images to the document
function addImagesToPdf(doc, before, after, gap, bW, bH, aW, aH, stacked, originalText) {
    var srCaption = 'Super-Resolved';

    if (sessionStorage.getItem('modelName')) {
        var srCaption = sessionStorage.getItem('modelName');
    }

    if (stacked == true) {
        doc.addImage(before.toDataURL(), 'PNG', gap, gap, bW, bH);
        doc.text(originalText, gap, gap + gap + bH);
        doc.addImage(after.toDataURL(), 'PNG', gap, gap + gap + gap + bH, aW, aH);
        doc.text(srCaption, gap, gap + gap + gap + gap + bH + aH);
    } else {
        doc.addImage(before.toDataURL(), 'PNG', gap, gap, bW, bH);
        doc.text(originalText, gap, gap + gap + bH);
        doc.addImage(after.toDataURL(), 'PNG', gap + gap + bW, gap, aW, aH);
        doc.text(srCaption, gap + gap + bW, gap + gap + aH);
    }
}

// Handler for the download button which gives the option to save the before-after images as PDF
function downloadComparisonPDF() {
    // FROM TESTING (with px_scaling, landscape):
    // 893px = 420mm (a3 width)
    // 1px = 0.470mm
    // 2.126px = 1mm
    // THUS:
    // 631px = 297mm (a3 heigth)

    var doc;
    var orientation;

    // Choose the PDF orientation based on the aspect ratio of the images
    if (comparisonCanvasBefore.width >= comparisonCanvasBefore.height) {
        orientation = 'l';
        var maxWidthA3 = 893;
        var maxHeightA3 = 631;
    } else {
        orientation = 'p';
        var maxWidthA3 = 631;
        var maxHeightA3 = 893;
    }

    doc = new jspdf.jsPDF(orientation, 'px', 'a3', true);

    // Gap between images themselves, margins and text
    var gap = 20

    var bW = croppedFrameCanvas.width;
    var bH = croppedFrameCanvas.height;
    var aW = comparisonCanvasAfter.width;
    var aH = comparisonCanvasAfter.height;

    // Add images in same page while keeping original size
    // Check if images can fit side by side
    if (((gap + gap + bW + aW) + gap) <= maxWidthA3) {
        var scaling = 1;

        // Check if the super-resolved image fits the height of the page and set a scaling if not possible
        if (((gap + gap + aH) + gap) > maxHeightA3) {
            scaling = aH / (maxHeightA3 - gap - gap - gap);
        }

        aW = aW / scaling;
        aH = aH / scaling;
        bW = bW / scaling;
        bH = bH / scaling;

        // Show images side-by-side
        addImagesToPdf(doc, croppedFrameCanvas, comparisonCanvasAfter, gap, bW, bH, aW, aH, false, 'Original');
    } else {
        var scaling = 1;

        // Check if the super-resolved image fits the width of the page and set a scaling if not possible
        if (((gap + aW) + gap) > maxWidthA3) {
            scaling = aW / (maxWidthA3 - gap - gap);
        }

        // Check if the images can fit stacked on the same page and set a scaling if not possible
        if (((gap + gap + gap + gap + bH + aH) + gap) > maxHeightA3) {
            var tempScaling = (bH + aH) / (maxHeightA3 - gap - gap - gap - gap - gap);
            if (tempScaling > scaling){
                scaling = tempScaling;
            }
        }

        aW = aW / scaling;
        aH = aH / scaling;
        bW = bW / scaling;
        bH = bH / scaling;

        // Show images stacked
        addImagesToPdf(doc, croppedFrameCanvas, comparisonCanvasAfter, gap, bW, bH, aW, aH, true, 'Original');
    }
    doc.addPage();

    // Reset the scaled dimensions
    bW = comparisonCanvasBefore.width;
    bH = comparisonCanvasBefore.height;
    aW = comparisonCanvasAfter.width;
    aH = comparisonCanvasAfter.height;

    // Add images side-by-side on same page while scaling the original to same size as the super-resolved
    // Check if their widths fit the page
    if (((gap + gap + aW + aW) + gap) <= maxWidthA3) {
        var scaling = 1;

        // Check if the super-resolved image fits the height of the page and set a scaling if not possible
        if (((gap + gap + aH) + gap) > maxHeightA3) {
            scaling = aH / (maxHeightA3 - gap - gap - gap);
        }

        aW = aW / scaling;
        aH = aH / scaling;
        bW = aW;
        bH = aH;
    } else {
        // Scale the widths of the images
        var scaling = (aW + aW) / (maxWidthA3 - gap - gap - gap);

        // Check if the super-resolved image fits the height of the page and re-set the scaling if not possible
        if (((gap + gap + aH) + gap) > maxHeightA3) {
            var tempScaling = aH / (maxHeightA3 - gap - gap - gap);
            if (tempScaling > scaling) {
                scaling = tempScaling;
            }
        }

        aW = aW / scaling;
        aH = aH / scaling;
        bW = aW;
        bH = aH;
    }
    // Show images side-by-side
    addImagesToPdf(doc, comparisonCanvasBefore, comparisonCanvasAfter, gap, bW, bH, aW, aH, false, 'Bicubic');
    doc.addPage();

    // Reset the scaled dimensions
    bW = comparisonCanvasBefore.width;
    bH = comparisonCanvasBefore.height;
    aW = comparisonCanvasAfter.width;
    aH = comparisonCanvasAfter.height;

    // Add images stacked on same page while scaling the original to same size as the super-resolved
    // Check if their heights fit the page
    if (((gap + gap + gap + gap + aH + aH) + gap) <= maxHeightA3) {
        var scaling = 1;

        // Check if the super-resolved image fits the width of the page and set a scaling if not possible
        if (((gap + aW) + gap) > maxWidthA3) {
            scaling = aW / (maxWidthA3 - gap - gap);
        }

        aW = aW / scaling;
        aH = aH / scaling;
        bW = aW;
        bH = aH;
    } else {
        // Scale the heights of the images
        var scaling = (aH + aH) / (maxHeightA3 - gap - gap - gap - gap - gap);

        // Check if the super-resolved image fits the widtth of the page and re-set the scaling if not possible
        if (((gap + aW) + gap) > maxWidthA3) {
            var tempScaling = aW / (maxWidthA3 - gap - gap);
            if (tempScaling > scaling) {
                scaling = tempScaling;
            }
        }

        aW = aW / scaling;
        aH = aH / scaling;
        bW = aW;
        bH = aH;
    }
    // Show images stacked
    addImagesToPdf(doc, comparisonCanvasBefore, comparisonCanvasAfter, gap, bW, bH, aW, aH, true, 'Bicubic');;

    doc.save(downloadNameComp);
}

function setSavedValues() {
    if (sessionStorage.getItem('models')) {
        // TODO: Add functionality for this
    }

    if (sessionStorage.getItem('processor')) {
        // TODO: Add functionality for this
    }

    if (sessionStorage.getItem('metadata')) {
        if (sessionStorage.getItem('metadata') == 'yes') {
            useMetadata = true;
        } else {
            useMetadata = false;
        }
    } else {
        useMetadata = false;
    }

    if (sessionStorage.getItem('url')) {
        apiLink = sessionStorage.getItem('url');
    } else {
        apiLink = 'http://127.0.0.1:5000/';
    }

    if (sessionStorage.getItem('videoFps')) {
        var chosenFps = sessionStorage.getItem('videoFps');
        frameTime = 1.0 / parseFloat(chosenFps);
    } else {
        frameTime = 0.03;
    }
}

// Set the adjustable values of the script
window.addEventListener('DOMContentLoaded', setSavedValues);

// Load the video/image from the user
content.addEventListener('change', loadContent);

// Load the metadata file from the user
metadata.addEventListener('change', loadedMetadata);

// Remove the default controls and make the custom controls visible
media.removeAttribute('controls');

// Pause and play the video when the button is clicked
play.addEventListener('click', playPauseMedia);

// Stop the video when stop is clicked or when the video has ended
stop.addEventListener('click', stopMedia);
media.addEventListener('ended', stopMedia);

// Fast forward and rewind the video
rwd.addEventListener('click', mediaBackward);
fwd.addEventListener('click', mediaForward);

// Frame forward and frame back
frameFwd.addEventListener('click', stepForward);
frameBck.addEventListener('click', stepBackward);

// Time update every time the video moves forward
media.addEventListener('timeupdate', setTime);

// Cut/choose a single frame from the video
frameCut.addEventListener('click', chooseFrame);

// Start, cancel and save crop of the chosen frame
crop.addEventListener('click', startCropping);
cancelcrop.addEventListener('click', cancelCropping);
save.addEventListener('click', saveCrop);

// Send the cropped image to the super-resolution API
superResolve.addEventListener('click', superResolveFrame);

// Download the super-resolved image
downloadSR.addEventListener('click', downloadSuperResolvedFrame);

// Show the before and after images
showComp.addEventListener('click', showComparisonCanvases);

// Download the comparison PDF
downloadComp.addEventListener('click', downloadComparisonPDF);
