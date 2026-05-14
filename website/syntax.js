const codeBlocks = document.querySelectorAll("pre[data-lang] code");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function highlightWithRules(source, rules) {
  const protectedTokens = [];
  let html = escapeHtml(source);

  function protect(pattern, className) {
    html = html.replace(pattern, (match) => {
      const token = `\uE000${String.fromCharCode(0xe100 + protectedTokens.length)}\uE001`;
      protectedTokens.push(`<span class="${className}">${match}</span>`);
      return token;
    });
  }

  for (const [pattern, className] of rules) {
    protect(pattern, className);
  }

  return html.replace(/\uE000([\uE100-\uE5FF])\uE001/g, (_, marker) => {
    return protectedTokens[marker.charCodeAt(0) - 0xe100];
  });
}

const rulesByLanguage = {
  shell: [
    [/^#.*$/gm, "tok-comment"],
    [/--[a-z0-9-]+/gi, "tok-flag"],
    [/\b(controlkit|python|pytest|cd|git)\b/g, "tok-command"],
    [/\b(validate|inspect|compile|benchmark|install|clone)\b/g, "tok-keyword"],
    [/\b(build\/[^\s]+|examples\/[^\s]+|custom_room_lqr\.yaml)\b/g, "tok-path"],
    [/\b\d+(?:\.\d+)?\b/g, "tok-number"],
  ],
  yaml: [
    [/^#.*$/gm, "tok-comment"],
    [/"[^"\n]*"|'[^'\n]*'/g, "tok-string"],
    [/^\s*[A-Za-z_][A-Za-z0-9_]*:/gm, "tok-key"],
    [/\b(lqr|mpc|rl)\b/g, "tok-keyword"],
    [/-?\b\d+(?:\.\d+)?\b/g, "tok-number"],
  ],
  json: [
    [/"(?:[^"\\]|\\.)*"\s*:/g, "tok-key"],
    [/"(?:[^"\\]|\\.)*"/g, "tok-string"],
    [/\b\d+(?:\.\d+)?\b/g, "tok-number"],
  ],
  python: [
    [/^#.*$/gm, "tok-comment"],
    [/"[^"\n]*"|'[^'\n]*'/g, "tok-string"],
    [/\b(for|in|range|if|else|return|lambda)\b/g, "tok-keyword"],
    [/\b(IRModule|RlPolicyIR|clip|activation|vector|matvec)\b/g, "tok-type"],
    [/-?\b\d+(?:\.\d+)?\b/g, "tok-number"],
  ],
  c: [
    [/^#.*$/gm, "tok-comment"],
    [/\/\/.*$/gm, "tok-comment"],
    [/"[^"\n]*"/g, "tok-string"],
    [/\b(void|const|float|int|for|return)\b/g, "tok-keyword"],
    [/\b(CONTROLKIT_[A-Z_]+)\b/g, "tok-type"],
    [/-?\b\d+(?:\.\d+)?f?\b/g, "tok-number"],
  ],
  rust: [
    [/\/\/.*$/gm, "tok-comment"],
    [/#!\[[^\]]+\]/g, "tok-flag"],
    [/"[^"\n]*"/g, "tok-string"],
    [/\b(pub|fn|mut|const|let|for|in)\b/g, "tok-keyword"],
    [/\b(f32|usize)\b/g, "tok-type"],
    [/-?\b\d+(?:\.\d+)?\b/g, "tok-number"],
  ],
  text: [
    [/^#.*$/gm, "tok-comment"],
    [/-?\b\d+(?:\.\d+)?\b/g, "tok-number"],
  ],
};

for (const block of codeBlocks) {
  const language = block.parentElement.dataset.lang || "text";
  const rules = rulesByLanguage[language] || rulesByLanguage.text;
  block.innerHTML = highlightWithRules(block.textContent, rules);
}
