const root = document.documentElement;
const toggle = document.querySelector(".theme-toggle");
const storageKey = "controlkit-theme";
const moonIcon = `
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path d="M21 13.1A8.2 8.2 0 0 1 10.9 3a7.1 7.1 0 1 0 10.1 10.1Z" />
  </svg>
`;
const sunIcon = `
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2" />
    <path d="M12 20v2" />
    <path d="m4.93 4.93 1.41 1.41" />
    <path d="m17.66 17.66 1.41 1.41" />
    <path d="M2 12h2" />
    <path d="M20 12h2" />
    <path d="m6.34 17.66-1.41 1.41" />
    <path d="m19.07 4.93-1.41 1.41" />
  </svg>
`;

function readStoredTheme() {
  try {
    return window.localStorage.getItem(storageKey);
  } catch {
    return null;
  }
}

function writeStoredTheme(theme) {
  try {
    window.localStorage.setItem(storageKey, theme);
  } catch {
    return null;
  }
  return theme;
}

const storedTheme = readStoredTheme();

if (storedTheme === "light" || storedTheme === "dark") {
  root.dataset.theme = storedTheme;
}

function resolvedTheme() {
  if (root.dataset.theme === "dark" || root.dataset.theme === "light") {
    return root.dataset.theme;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function syncToggle() {
  const theme = resolvedTheme();
  const next = theme === "dark" ? "light" : "dark";
  toggle.innerHTML = next === "dark" ? moonIcon : sunIcon;
  toggle.setAttribute("aria-label", `Switch to ${next} mode`);
  toggle.setAttribute("title", `Switch to ${next} mode`);
}

toggle.addEventListener("click", () => {
  const next = resolvedTheme() === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  writeStoredTheme(next);
  syncToggle();
});

syncToggle();
