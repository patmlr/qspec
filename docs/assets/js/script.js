
window.addEventListener("beforeunload", () => {
  localStorage.setItem("scrollPositon", document.querySelector(".bd-content-table-content").scrollTop);
});

window.addEventListner("load", () => {
  document.querySelector(".bd-content-table-content").scrollTop = localStorage.getItem("scrollPositon") || 0;
});


