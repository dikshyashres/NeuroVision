document.addEventListener("DOMContentLoaded", () => {
  console.log("Auth JS Loaded");

  const loginForm = document.getElementById("login");
  const registerForm = document.getElementById("register");

  // ---------------- LOGIN ----------------
  loginForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const email = loginForm.querySelector('input[type="text"]').value.trim();
    const password = loginForm
      .querySelector('input[type="password"]')
      .value.trim();

    if (email === "" || password === "") {
      alert("❌ Please fill all fields");
      return;
    }

    alert("✅ Login Successful");

    // 🔁 Redirect to Detection Page
    window.location.href = "detection.html";
  });

  // ---------------- REGISTER ----------------
  registerForm.addEventListener("submit", (e) => {
    e.preventDefault();

    const inputs = registerForm.querySelectorAll("input");
    const password = inputs[2].value;
    const confirmPassword = inputs[3].value;

    if (password !== confirmPassword) {
      alert("❌ Passwords do not match");
      return;
    }

    alert("✅ Registration Successful");

    // Switch back to login page
    document.querySelector('a[href="#login"]').click();
  });
});
