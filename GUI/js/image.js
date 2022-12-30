function previewFile() {
    let preview = document.querySelector('img'); // Get HTML img element
    let file    = document.querySelector('input[type=file]').files[0]; // Get HTML file element (interface to allow user to select a file from disk)
    let reader  = new FileReader(); // Create new file reader
    // When the user has chosen a file
    reader.onloadend = function () {
        // Set the chosen image to be displayed
        let sourceImage = reader.result;
        preview.src = sourceImage;
        // Get the base64 representation of the chosen image by removing initial string that is identical across images
        let base64Img = sourceImage.substr(22);

        let apiUrl = "http://127.0.0.1:5000/";
        if (sessionStorage.getItem('url')) {
            apiUrl = sessionStorage.getItem('url');
        }

        $.post(apiUrl + "super_resolve",  {'image': base64Img}).done(function(data) {
            let preview = document.getElementById('super_resolved_im');  // set base64 image to display
            preview.src = 'data:image/png;base64,'.concat(data);
            let load_img_cont = document.getElementById("super_resolved");
            load_img_cont.style.display = "block";
        });
    }

    if (file) {
        reader.readAsDataURL(file); // Read image
        let load_img_cont = document.getElementById("loaded_img");
        load_img_cont.style.display = "block";   // Show the container with image
    } else {
        preview.src = "";
    }
}
