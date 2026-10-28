// Login and register pages: the form, and the "agent at work" panel.
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  // ---- the agent at work (decoration: it shows what MeetPilot does) -------------------------------
  const steps = [
    ["📧", "New mail arrives", "“Client review at BIEC”"],
    ["🧠", "Reading the email", "in person · 20 Oct · 11:00"],
    ["📍", "Finding the venue", "Bangalore International Exhibition Centre"],
    ["🚗", "Checking the route", "432 km · about 7 h 35 min"],
    ["🏨", "Searching hotels", "4 within your budget"],
    ["🔒", "Waiting for your approval", ""],
  ];

  const list = $(".run-steps");
  const actions = $(".run-actions");
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function mark(kind, text) {
    const m = document.createElement("span");
    m.className = "mark " + kind;
    if (text) m.textContent = text;
    return m;
  }

  if (list) {
    const rows = steps.map(([icon, title, detail]) => {
      const li = document.createElement("li");
      const b = document.createElement("b");
      b.textContent = title;
      li.append(mark("spin"), icon + " ", b);
      if (detail) li.append(" · " + detail);
      list.appendChild(li);
      return li;
    });

    const finish = (li, last) => {
      li.classList.add("done");
      li.firstChild.replaceWith(last ? mark("lock", "🔒") : mark("tick", "✓"));
    };

    if (reduced) {
      rows.forEach((li, i) => { li.classList.add("on"); finish(li, i === rows.length - 1); });
      if (actions) actions.classList.add("on");
    } else {
      let i = 0;

      const tick = () => {
        if (i > 0) finish(rows[i - 1], false);

        if (i < rows.length) {
          rows[i].classList.add("on");
          if (i === rows.length - 1) {
            rows[i].firstChild.replaceWith(mark("lock", "🔒"));
            if (actions) actions.classList.add("on");
          }
          i += 1;
          setTimeout(tick, i === rows.length ? 1800 : 1150);
        } else {
          // start over after a pause
          setTimeout(() => {
            rows.forEach((li) => {
              li.className = "";
              li.firstChild.replaceWith(mark("spin"));
            });
            if (actions) actions.classList.remove("on");
            i = 0;
            tick();
          }, 3200);
        }
      };

      tick();
    }
  }

  // ---- the form -----------------------------------------------------------------------------------------------
  const form = document.querySelector("form[data-auth]");
  if (!form) return;

  const mode = form.dataset.auth;                 // "login" or "register"
  const errorBox = $("#error");
  const button = $("#submit");
  const buttonText = button.textContent;
  const emailInput = $("#email");
  const passwordInput = $("#password");

  function showError(message, field) {
    errorBox.textContent = "";
    const icon = document.createElement("span");
    icon.textContent = "⚠️";
    errorBox.append(icon, document.createTextNode(message));
    errorBox.hidden = false;

    [emailInput, passwordInput].forEach((el) => el && el.setAttribute("aria-invalid", "false"));
    const target = field === "email" ? emailInput : field === "password" ? passwordInput : null;

    if (target) {
      target.setAttribute("aria-invalid", "true");
      target.focus();
    }
  }

  function clearError() {
    errorBox.hidden = true;
    [emailInput, passwordInput].forEach((el) => el && el.setAttribute("aria-invalid", "false"));
  }

  function busy(on) {
    button.disabled = on;
    button.textContent = "";

    if (on) {
      button.append(mark("spin"), document.createTextNode(mode === "login" ? "Signing in…" : "Creating your account…"));
    } else {
      button.textContent = buttonText;
    }
  }

  // show / hide the password
  document.querySelectorAll("[data-toggle-password]").forEach((toggle) => {
    toggle.addEventListener("click", () => {
      const input = document.getElementById(toggle.dataset.togglePassword);
      const show = input.type === "password";
      input.type = show ? "text" : "password";
      toggle.textContent = show ? "Hide" : "Show";
      toggle.setAttribute("aria-pressed", String(show));
    });
  });

  // how strong the new password is (the server has the final say)
  const meter = $(".meter");
  const hint = $("#password-hint");

  function strength(pw) {
    if (!pw) return { level: 0, text: "Use 12 or more characters. A short phrase of a few words works well." };
    if (pw.length < 12) return { level: 1, text: (12 - pw.length) + " more character" + (12 - pw.length === 1 ? "" : "s") + " to go." };

    let score = 2;
    if (pw.length >= 16) score += 1;
    if (/\s/.test(pw) || (/[a-z]/.test(pw) && /[A-Z]/.test(pw) && /\d/.test(pw)) || /[^A-Za-z0-9]/.test(pw)) score += 1;

    return { level: Math.min(score, 4), text: score >= 4 ? "Strong." : score === 3 ? "Good. Longer is stronger." : "Okay. Add more words to make it stronger." };
  }

  if (meter) {
    const update = () => {
      const s = strength(passwordInput.value);
      meter.dataset.level = String(s.level);
      hint.textContent = s.text;
    };
    passwordInput.addEventListener("input", update);
    update();
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();

    const email = emailInput.value.trim();
    const password = passwordInput.value;

    if (!/^[^@\s]+@[^@\s]+\.[^@\s]{2,}$/.test(email)) return showError("Enter a valid email address.", "email");
    if (!password) return showError("Enter your password.", "password");
    if (mode === "register" && password.length < 12) return showError("Use at least 12 characters. A short phrase of a few words works well.", "password");

    busy(true);

    try {
      const body = { email, password };
      if (mode === "register") body.name = ($("#name").value || "").trim();

      const res = await fetch("/auth/" + mode, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-MeetPilot": "1" },
        body: JSON.stringify(body),
        credentials: "same-origin",
      });

      let data = {};
      try { data = await res.json(); } catch (e) { /* not JSON */ }

      if (res.ok) {
        window.location.href = "/";
        return;
      }

      busy(false);
      showError(data.error || "Something went wrong (" + res.status + "). Please try again.", data.field);
    } catch (err) {
      busy(false);
      showError("Can't reach the server. Check that MeetPilot is running, then try again.");
    }
  });

  (emailInput || passwordInput).focus();
})();
