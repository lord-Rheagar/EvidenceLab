import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import '@fontsource/ibm-plex-mono/latin-600.css';
import '@fontsource/ibm-plex-mono/latin-700.css';
import '@fontsource/inter/latin-400.css';
import '@fontsource/inter/latin-500.css';
import '@fontsource/inter/latin-600.css';
import '@fontsource/inter/latin-700.css';
import { createIcons, Terminal, GitBranch, ArrowUpRight, ArrowDown, Copy, Check, Search, FileText, RotateCcw, CheckCircle2, Link, Quote, Download, Settings2, ScanSearch, History, Layers, Braces, Sparkles, Asterisk, Star, Network, Plug, Plus } from 'lucide';

createIcons({ icons: { Terminal, GitBranch, ArrowUpRight, ArrowDown, Copy, Check, Search, FileText, RotateCcw, CheckCircle2, Link, Quote, Download, Settings2, ScanSearch, History, Layers, Braces, Sparkles, Asterisk, Star, Network, Plug, Plus } });
const status = document.querySelector('#copy-status');
let statusTimeout;
function notify(message) {
  clearTimeout(statusTimeout);
  status.textContent = message;
  status.classList.add('is-visible');
  statusTimeout = setTimeout(() => status.classList.remove('is-visible'), 2600);
}
async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return; } catch { /* Try the local-file fallback. */ }
  }
  const field = document.createElement('textarea');
  field.value = text;
  field.setAttribute('readonly', '');
  field.style.cssText = 'position:fixed;left:-9999px;top:0';
  document.body.append(field);
  const focused = document.activeElement;
  try { field.select(); if (!document.execCommand('copy')) throw new Error('Copy unavailable'); }
  finally { field.remove(); focused?.focus(); }
}
document.querySelectorAll('.copy').forEach(button => {
  button.addEventListener('click', async () => {
    button.disabled = true;
    try { await copyText(button.dataset.copy); notify('Command copied.'); }
    catch { notify('Copy unavailable. Select the command and copy it.'); }
    finally { button.disabled = false; }
  });
});
const tabs = [...document.querySelectorAll('[role="tab"]')];
const stages = [...document.querySelectorAll('[data-stage]')];
const result = document.querySelector('.terminal-result');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let stageTimeouts = [];
function stopPreview() { stageTimeouts.forEach(clearTimeout); stageTimeouts = []; }
function animateResearch() {
  stopPreview();
  stages.forEach(stage => stage.classList.toggle('stage-done', reducedMotion.matches));
  result.style.visibility = reducedMotion.matches ? 'visible' : 'hidden';
  if (reducedMotion.matches) return;
  stages.forEach((stage, index) => stageTimeouts.push(setTimeout(() => stage.classList.add('stage-done'), index * 400 + 100)));
  stageTimeouts.push(setTimeout(() => { result.style.visibility = 'visible'; }, 2100));
}
function showTab(name, focus = false) {
  stopPreview();
  tabs.forEach(tab => {
    const active = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
    if (active && focus) tab.focus();
  });
  if (name === 'research') animateResearch();
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => showTab(tab.dataset.tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next !== undefined) { event.preventDefault(); showTab(tabs[next].dataset.tab, true); }
  });
});
document.querySelector('.replay').addEventListener('click', () => showTab('research'));
reducedMotion.addEventListener('change', () => {
  if (document.querySelector('#tab-research').getAttribute('aria-selected') === 'true') animateResearch();
});
const environmentNames = { openai:'OPENAI_API_KEY', anthropic:'ANTHROPIC_API_KEY', gemini:'GOOGLE_API_KEY', openrouter:'OPENROUTER_API_KEY', compatible:'EVIDENCE_CUSTOM_API_KEY' };
document.querySelectorAll('[data-provider]').forEach(button => {
  button.addEventListener('click', () => {
    const provider = button.dataset.provider;
    document.querySelectorAll('[data-provider]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    document.querySelector('#provider-name').textContent = provider;
    document.querySelector('#provider-env').textContent = environmentNames[provider];
    document.querySelectorAll('.selected-provider').forEach(item => { item.textContent = provider; });
  });
});

