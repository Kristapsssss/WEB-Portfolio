// Toggle Dash iframe visibility

function toggleIframe(id) {
    var iframe = document.getElementById(id);
    if (iframe.style.display === "none" || iframe.style.display === "") {
        iframe.style.display = "block";
    } else {
        iframe.style.display = "none";
    }
}


