$(document).ready(function() {
    // Handle input for search suggestions
    $('#search-box').on('input', function() {
        let query = $(this).val();
        console.log("Sending query:", query);  // Log the query
        if (query.length > 0) {
          $.get('/search_animes', { query: query }, function(data) {
              $('#search-results').empty();  // Clear previous results
              data.forEach(item => {
                  $('#search-results').append(`<li class="list-group-item" data-title="${item.Title}">${item.Title} (${item.Release})</li>`);
              });
          });
      } else {
          $('#search-results').empty();  // Clear results if query is empty
      }
  });
  
    let selectedAnimes = []; // Declare the array to store selected animes

    // Add a click event for the search suggestions
    $('#search-results').on('click', 'li', function () {
    let selectedTitle = $(this).data('title');

    if (selectedAnimes.includes(selectedTitle)) {
        // Show notification for already selected anime
        Swal.fire({
            icon: 'warning',
            title: 'Already Selected',
            text: `"${selectedTitle}" is already in your list!`,
            showConfirmButton: false,
            timer: 1500,
            toast: true,
            position: 'top-end',
        });
    } else if (selectedAnimes.length >= 5) {
        // Show notification for exceeding limit
        Swal.fire({
            icon: 'error',
            title: 'Limit Reached',
            text: 'You can only select up to 5 animes!',
            showConfirmButton: false,
            timer: 1500,
            toast: true,
            position: 'top-end',
        });
    } else {
        // Add the anime to the selected list
        selectedAnimes.push(selectedTitle);
        updateSelectedAnimes();

        // Show success notification
        Swal.fire({
            icon: 'success',
            title: 'Anime Added',
            text: `"${selectedTitle}" has been added to your list!`,
            showConfirmButton: false,
            timer: 1500,
            toast: true,
            position: 'top-end',
        });
    }
});
    
    // Function to update the displayed selected animes
    function updateSelectedAnimes() {
        $('#selected-animes').empty();  // Clear the current selected animes
        selectedAnimes.forEach(title => {
            $('#selected-animes').append(`<li class="list-group-item d-flex justify-content-between align-items-center">${title} <button class="remove-btn" data-title="${title}">Remove</button></li>`);
        });
    }

    // Remove an anime from the selected list
    $('#selected-animes').on('click', '.remove-btn', function() {
        let titleToRemove = $(this).data('title');
        selectedAnimes = selectedAnimes.filter(title => title !== titleToRemove);  // Remove title from the list
        updateSelectedAnimes();  // Update the displayed list
    });

    // Handle the "Get Recommendations" button (if you want to send the selected animes for recommendations)
    $('#recommend-button').on('click', function () {
    if (selectedAnimes.length === 0) {
        // Show notification if no anime is selected
        Swal.fire({
            icon: 'error',
            title: 'No Anime Selected',
            text: 'Please select at least one anime before getting recommendations!',
            showConfirmButton: false,
            timer: 2000,
            toast: true,
            position: 'top-end',
        });
        return; // Exit the function early
    }

    function displayRecommendations(recommendations) {
const recommendationsContainer = $('#recommendations-container');
recommendationsContainer.empty(); // Clear previous recommendations

recommendations.forEach((anime) => {
recommendationsContainer.append(`
    <div class="anime-card card my-3 shadow-sm" style="border-radius: 15px;">
        <div class="card-body">
            <h5 class="card-title text-primary font-weight-bold">${anime.Title}</h5>
            <h6 class="card-subtitle mb-2 text-success">Score: ${anime.Score}</h6>
            <p class="card-text">
                <strong>Release:</strong> ${anime.Release} <br>
                <strong>Episodes:</strong> ${anime.Episodes} <br>
                <strong>Genre:</strong> ${anime.Genre} <br>
                <strong>Synopsis:</strong> ${anime.Synopsis} <br>
                <strong>Popularity:</strong> ${Math.round(anime.Popularity)} <br>
                <strong>Demographic:</strong> ${anime.Demographic} <br>
                <strong>Studio:</strong> ${anime.Studio} <br>
                <strong>Theme:</strong> ${anime.Theme}
            </p>
        </div>
    </div>
`);
});
}


        // Send selected titles to the backend
        $.ajax({
            url: '/recommendations',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ titles: selectedAnimes }),
            success: function(response) {
                displayRecommendations(response);
            },
            error: function(xhr, status, error) {
                console.error("Error fetching recommendations:", error);
                alert("Failed to fetch recommendations. Please try again.");
            }
        });
    });
});

    // JS for Discover Section
$('#filter-button').on('click', function() {
    const sortBy = $('#sort-options').val();
    const filterYear = $('#year-filter').val();

    // Example logic for populating the Discover section
    $.ajax({
        url: '/discover_animes',
        type: 'GET',
        data: { sortBy, filterYear },
        success: function(data) {
            const container = $('#discover-container');
            container.empty(); // Clear previous results

            data.forEach(anime => {
                container.append(`
                    <div class="anime-card">
                        <h3>${anime.Title} (${anime.Release})</h3>
                        <p><strong>Score:</strong> ${anime.Score}</p>
                        <p><strong>Episodes:</strong> ${anime.Episodes}</p>
                        <p><strong>Genre:</strong> ${anime.Genre}</p>
                        <p><strong>Popularity:</strong> ${anime.Popularity}</p>
                    </div>
                `);
            });
        }
    });
});
