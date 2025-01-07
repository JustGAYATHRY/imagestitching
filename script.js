// Wait for the DOM to be fully loaded before executing any code
document.addEventListener('DOMContentLoaded', function() {
    // Function to stitch images
    function stitchImages(image1, image2) {
        // Here you would use OpenCV.js or another library to stitch the images
        // This is a simplified example assuming you have a function stitch that stitches two images
        // Replace this with actual stitching logic

        // For demonstration, just concatenate the two images horizontally
        let canvas = document.createElement('canvas');
        let ctx = canvas.getContext('2d');
        canvas.width = image1.width + image2.width;
        canvas.height = Math.max(image1.height, image2.height);
        ctx.drawImage(image1, 0, 0);
        ctx.drawImage(image2, image1.width, 0);

        // Convert canvas to a data URL
        let dataURL = canvas.toDataURL('image/png');

        // Store the data URL in localStorage to pass to the next page
        localStorage.setItem('stitchedImage', dataURL);

        // Redirect to the next page
        window.location.href = 'stitched.html';
    }

    // Function to handle image upload
    function handleImageUpload(event) {
        const file1 = document.getElementById('file1').files[0];
        const file2 = document.getElementById('file2').files[0];

        if (file1 && file2) {
            // Create image elements
            let image1 = new Image();
            let image2 = new Image();

            // Load images
            image1.onload = function() {
                image2.onload = function() {
                    // Call the stitchImages function with the loaded images
                    stitchImages(image1, image2);
                };
                image2.src = URL.createObjectURL(file2);
            };
            image1.src = URL.createObjectURL(file1);
        } else {
            alert('Please select two images.');
        }
    }

    // Attach handleImageUpload function to the button click event
    document.querySelector('button').addEventListener('click', handleImageUpload);
});