const slider = document.getElementById('comparison');
slider.addEventListener('input', () => {
  slider.closest('.comparison').style.setProperty('--split', `${slider.value}%`);
});
document.querySelectorAll('[data-scene]').forEach(button => {
  button.addEventListener('click', () => {
    const scene = button.dataset.scene;
    const label = button.textContent.slice(3);
    const result = document.querySelector('.compare-base');
    const input = document.querySelector('.compare-before img');
    result.src = `assets/${scene}-result.png`;
    result.alt = `GlassGS rendered result for the ${label.toLowerCase()} scene`;
    input.src = `assets/${scene}-input.png`;
    input.alt = `Captured input for the ${label.toLowerCase()} scene with glass reflections`;
    document.getElementById('scene-caption').textContent = `${button.textContent.slice(0, 2)} — ${label} / real-world reflective enclosure`;
    document.querySelectorAll('[data-scene]').forEach(item => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    slider.value = '50';
    slider.closest('.comparison').style.setProperty('--split', '50%');
  });
});
const tabs = Array.from(document.querySelectorAll('[data-panel]'));
function selectTab(button) {
  tabs.forEach(item => {
    const active = item === button;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
    document.getElementById(item.dataset.panel).hidden = !active;
  });
}
tabs.forEach((button, index) => {
  button.addEventListener('click', () => selectTab(button));
  button.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    selectTab(tabs[next]);
    tabs[next].focus();
  });
});
let toastTimer;
document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copy);
    const toast = document.getElementById('toast');
    try {
      await navigator.clipboard.writeText(source.textContent);
      toast.textContent = 'Copied to clipboard';
    } catch (error) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(source);
      selection.removeAllRanges();
      selection.addRange(range);
      toast.textContent = 'Text selected — press Ctrl+C or ⌘C to copy';
    }
    clearTimeout(toastTimer);
    toast.classList.add('visible');
    toastTimer = setTimeout(() => toast.classList.remove('visible'), 3000);
  });
});
