/*
 * Minimal, safe Markdown renderer for LLM answers.
 * Security model: the input is HTML-escaped FIRST, then a small set of
 * Markdown constructs is turned into tags we generate ourselves. Links are
 * restricted to http(s). No raw HTML from the model ever reaches the DOM.
 * Supports: paragraphs, headings, bold/italic, inline code, ordered and
 * unordered lists, pipe tables, links, and citation markers like [3].
 */
(function (global) {
  "use strict";

  const escapeHtml = (s) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
     .replace(/"/g, "&quot;").replace(/'/g, "&#39;");

  function inline(text, sources) {
    let s = escapeHtml(text);
    const codes = [];
    s = s.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      (_, label, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`);
    // citations: [1], [1, 2], [1][2]
    s = s.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, (m, nums) =>
      nums.split(/\s*,\s*/).map((n) => {
        const src = sources && sources[Number(n) - 1];
        if (!src) return `[${n}]`;
        return `<a class="cite" href="${escapeHtml(src.url)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(src.title)}">${n}</a>`;
      }).join(""));
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
    s = s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
    return s;
  }

  const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

  function render(md, sources) {
    const lines = (md || "").replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let para = [];
    let list = null; // {type, items}

    const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join(" "), sources)}</p>`); para = []; } };
    const flushList = () => {
      if (list) { out.push(`<${list.type}>${list.items.map((i) => `<li>${inline(i, sources)}</li>`).join("")}</${list.type}>`); list = null; }
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!line.trim()) { flushPara(); flushList(); continue; }

      if (isTableRow(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        flushPara(); flushList();
        const head = cells(line);
        const rows = [];
        i += 2;
        while (i < lines.length && isTableRow(lines[i])) { rows.push(cells(lines[i])); i++; }
        i--;
        out.push('<div class="table-wrap"><table><thead><tr>' +
          head.map((h) => `<th>${inline(h, sources)}</th>`).join("") + "</tr></thead><tbody>" +
          rows.map((r) => "<tr>" + r.map((c) => `<td>${inline(c, sources)}</td>`).join("") + "</tr>").join("") +
          "</tbody></table></div>");
        continue;
      }

      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) { flushPara(); flushList(); const lvl = Math.min(h[1].length + 2, 4); out.push(`<h${lvl}>${inline(h[2], sources)}</h${lvl}>`); continue; }

      const ul = line.match(/^\s*[-*•]\s+(.*)$/);
      const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (ul || ol) {
        flushPara();
        const type = ul ? "ul" : "ol";
        if (!list || list.type !== type) { flushList(); list = { type, items: [] }; }
        list.items.push((ul || ol)[1]);
        continue;
      }
      if (list && /^\s{2,}\S/.test(line)) { list.items[list.items.length - 1] += " " + line.trim(); continue; }

      flushList();
      para.push(line.trim());
    }
    flushPara(); flushList();
    return out.join("");
  }

  global.SafeMarkdown = { render, escapeHtml };
})(window);
