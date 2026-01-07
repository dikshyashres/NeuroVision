// Select elements
const loginTab = document.querySelector('a[href="#login"]');
const registerTab = document.querySelector('a[href="#register"]');

const loginForm = document.getElementById("login");
const registerForm = document.getElementById("register");

// Function to switch forms
function showLogin() {
  loginForm.classList.add("active");
  registerForm.classList.remove("active");

  loginTab.classList.add("active");
  registerTab.classList.remove("active");
}

function showRegister() {
  registerForm.classList.add("active");
  loginForm.classList.remove("active");

  registerTab.classList.add("active");
  loginTab.classList.remove("active");
}

// Tab click events
loginTab.addEventListener("click", (e) => {
  e.preventDefault();
  showLogin();
});

registerTab.addEventListener("click", (e) => {
  e.preventDefault();
  showRegister();
});

// Switch links inside forms
document.querySelectorAll(".switch-form").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    if (link.getAttribute("href") === "#register") {
      showRegister();
    } else {
      showLogin();
    }
  });
});

// Login form submit
loginForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const email = loginForm.querySelector('input[type="text"]').value;
  const password = loginForm.querySelector('input[type="password"]').value;

  if (email && password) {
    alert("✅ Login Successful!");
    // redirect example:
    // window.location.href = "dashboard.html";
  } else {
    alert("❌ Please fill all fields");
  }
});

// Register form submit
registerForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const inputs = registerForm.querySelectorAll("input");
  const password = inputs[2].value;
  const confirmPassword = inputs[3].value;

  if (password !== confirmPassword) {
    alert("❌ Passwords do not match");
    return;
  }

  alert("✅ Registration Successful!");
  showLogin();
});
// Login form submit
loginForm.addEventListener("submit", (e) => {
  e.preventDefault();

  const email = loginForm.querySelector('input[type="text"]').value;
  const password = loginForm.querySelector('input[type="password"]').value;

  if (email && password) {
    alert("✅ Login Successful!");

    // 🔁 Redirect to detection page
    window.location.href = "detection.html";
  } else {
    alert("❌ Please fill all fields");
  }
});
