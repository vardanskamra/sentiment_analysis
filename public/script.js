$(document).ready(function() {
    $('#analyzeButton').click(function() {
        const tweetInput = $('#tweetInput').val().trim();

        if (!tweetInput) {
            alert('Please enter a tweet.');
            return;
        }

        document.getElementById('result').innerText = "Please Wait...";

        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tweet: tweetInput })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text();
        })
        .then(result => {
            // Blink background color based on result
            if (result === 'Positive') {
                blinkBackground('#83ed55', 3);
            } else if(result == 'Neutral'){
                blinkBackground('#ede855', 3)
            } else {
                blinkBackground('#f53d3d', 3);
            }

            document.getElementById('result').innerText = result;
        })
        .catch(error => {
            console.error('Error occurred:', error);
        });
    });
});

function blinkBackground(color, times) {
    let count = 0;
    const interval = setInterval(function() {
        document.body.style.backgroundColor = color;
        count++;
        if (count === times * 2) {
            clearInterval(interval);
            document.body.style.backgroundColor = ''; // Reset to default color
        }
    }, 500); // Blink every 500 milliseconds
}
