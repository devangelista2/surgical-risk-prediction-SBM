/* Copied from neurosurg-predict v1.5.0: applies the saved theme before first paint, "system" leaving the
   attribute off so the media query decides. */
(() => {
  const KEY = 'theme';
  const root = document.documentElement;

  function choice() {
    try { const v = localStorage.getItem(KEY); return v === 'light' || v === 'dark' ? v : 'system'; }
    catch (e) { return 'system'; }
  }

  function paint(value) {
    if (value === 'system') delete root.dataset.theme;
    else root.dataset.theme = value;
  }

  paint(choice());
  root.dataset.themeReady = '';

  document.addEventListener('DOMContentLoaded', () => {
    const picked = choice();
    document.querySelectorAll('input[name="theme"]').forEach(radio => {
      radio.checked = radio.value === picked;
      radio.addEventListener('change', () => {
        paint(radio.value);
        try {
          if (radio.value === 'system') localStorage.removeItem(KEY);
          else localStorage.setItem(KEY, radio.value);
        } catch (e) { /* private mode refuses to store: the choice still holds for this page */ }
      });
    });
  });
})();
